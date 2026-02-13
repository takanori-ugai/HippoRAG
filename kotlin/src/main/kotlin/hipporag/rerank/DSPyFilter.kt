package hipporag.rerank

import hipporag.llm.BaseLLM
import hipporag.prompts.BEST_DSPY_PROMPT_JSON
import hipporag.utils.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

class DSPyFilter(
    private val llmModel: BaseLLM,
    private val rerankDspyFilePath: String? = null,
) {
    private val logger = KotlinLogging.logger {}
    private val json = Json { ignoreUnknownKeys = true }
    private val messageTemplate: List<Message> = buildTemplate()

    private val oneInputTemplate =
        """
        [[ ## question ## ]]
        {question}

        [[ ## fact_before_filter ## ]]
        {fact_before_filter}

        Respond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid JSON), and then ending with the marker for `[[ ## completed ## ]]`.
        """.trimIndent()

    private val oneOutputTemplate =
        """
        [[ ## fact_after_filter ## ]]
        {fact_after_filter}

        [[ ## completed ## ]]
        """.trimIndent()

    fun rerank(
        query: String,
        candidateFacts: List<List<String>>,
        candidateFactIndices: List<Int>,
        lenAfterRerank: Int,
    ): Triple<List<Int>, List<List<String>>, Map<String, Any?>> {
        if (candidateFacts.isEmpty()) {
            return Triple(emptyList(), emptyList(), mapOf("facts_after_rerank" to emptyList<List<String>>()))
        }

        val factBeforeFilter =
            JsonObject(
                mapOf(
                    "fact" to
                        JsonArray(
                            candidateFacts.map { fact ->
                                JsonArray(fact.map { JsonPrimitive(it) })
                            },
                        ),
                ),
            )

        val userPrompt =
            oneInputTemplate
                .replace("{question}", query)
                .replace("{fact_before_filter}", json.encodeToString(JsonObject.serializer(), factBeforeFilter))

        val messages = messageTemplate + Message(role = "user", content = userPrompt)

        return runCatching {
            val result = llmModel.infer(messages)
            val filteredFacts = parseFacts(result.response)
            val (sortedIndices, sortedFacts) =
                matchFactsToCandidates(filteredFacts, candidateFacts, candidateFactIndices)
            Triple(
                sortedIndices.take(lenAfterRerank),
                sortedFacts.take(lenAfterRerank),
                mapOf("model_response" to result.response, "confidence" to null),
            )
        }.getOrElse { e ->
            logger.warn(e) { "DSPy rerank failed, falling back to original order." }
            Triple(
                candidateFactIndices.take(lenAfterRerank),
                candidateFacts.take(lenAfterRerank),
                mapOf("error" to e.message.orEmpty(), "confidence" to null),
            )
        }
    }

    private fun parseFacts(response: String): List<List<String>> {
        val jsonObj = extractJsonObjectWithKey(response, "fact") ?: return emptyList()
        val factArray = jsonObj["fact"]?.jsonArray ?: return emptyList()
        val result = mutableListOf<List<String>>()
        for (factEl in factArray) {
            val triple = runCatching { factEl.jsonArray }.getOrNull() ?: continue
            val tripleList = triple.mapNotNull { it.jsonPrimitive.contentOrNull?.trim() }.filter { it.isNotEmpty() }
            if (tripleList.isNotEmpty()) {
                result.add(tripleList)
            }
        }
        return result
    }

    private fun extractJsonObjectWithKey(
        response: String,
        key: String,
    ): JsonObject? {
        val pattern = Regex("\\{[^{}]*\"$key\"\\s*:\\s*\\[[\\s\\S]*?\\][^{}]*\\}", RegexOption.DOT_MATCHES_ALL)
        val match = pattern.find(response) ?: return null
        return runCatching { json.parseToJsonElement(match.value).jsonObject }.getOrNull()
    }

    private fun matchFactsToCandidates(
        filteredFacts: List<List<String>>,
        candidateFacts: List<List<String>>,
        candidateFactIndices: List<Int>,
    ): Pair<List<Int>, List<List<String>>> {
        if (filteredFacts.isEmpty()) {
            return candidateFactIndices to candidateFacts
        }

        val indices = mutableListOf<Int>()
        val facts = mutableListOf<List<String>>()

        for (fact in filteredFacts) {
            val exactIdx = candidateFacts.indexOfFirst { it == fact }
            val idx =
                if (exactIdx >= 0) {
                    exactIdx
                } else {
                    bestFuzzyMatchIndex(fact, candidateFacts)
                }
            if (idx >= 0) {
                indices.add(candidateFactIndices[idx])
                facts.add(candidateFacts[idx])
            }
        }

        if (indices.isEmpty()) {
            return candidateFactIndices to candidateFacts
        }

        return indices to facts
    }

    private fun bestFuzzyMatchIndex(
        fact: List<String>,
        candidates: List<List<String>>,
    ): Int {
        val target = normalizeFact(fact)
        var bestIdx = -1
        var bestScore = 0.0
        for ((idx, candidate) in candidates.withIndex()) {
            val score = jaccardSimilarity(target, normalizeFact(candidate))
            if (score > bestScore) {
                bestScore = score
                bestIdx = idx
            }
        }
        return if (bestScore >= 0.2) bestIdx else -1
    }

    private fun normalizeFact(fact: List<String>): Set<String> =
        fact
            .joinToString(" ")
            .lowercase()
            .replace(Regex("[^a-z0-9 ]"), " ")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .toSet()

    private fun jaccardSimilarity(
        a: Set<String>,
        b: Set<String>,
    ): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val intersect = a.intersect(b).size.toDouble()
        val union = (a.size + b.size - intersect)
        return if (union == 0.0) 0.0 else intersect / union
    }

    private fun buildTemplate(): List<Message> {
        val jsonText = loadPromptJson()
        val root = runCatching { json.parseToJsonElement(jsonText).jsonObject }.getOrNull()
        val prog = root?.get("prog")?.jsonObject
        val system = prog?.get("system")?.jsonPrimitive?.contentOrNull
        val demos = prog?.get("demos")?.jsonArray ?: JsonArray(emptyList())

        val messages = mutableListOf<Message>()
        if (system != null) {
            messages.add(Message(role = "system", content = system))
        } else {
            messages.add(
                Message(
                    role = "system",
                    content = "Filter facts relevant to the question. Output JSON only.",
                ),
            )
        }

        for (demo in demos) {
            val obj = demo.jsonObject
            val question = obj["question"]?.jsonPrimitive?.contentOrNull
            val factBefore = obj["fact_before_filter"]?.jsonPrimitive?.contentOrNull
            val factAfter = obj["fact_after_filter"]?.jsonPrimitive?.contentOrNull
            if (question == null || factBefore == null || factAfter == null) continue

            val userMsg =
                oneInputTemplate
                    .replace("{question}", question)
                    .replace("{fact_before_filter}", factBefore)
            val assistantMsg =
                oneOutputTemplate
                    .replace("{fact_after_filter}", factAfter)

            messages.add(Message(role = "user", content = userMsg))
            messages.add(Message(role = "assistant", content = assistantMsg))
        }

        return messages
    }

    private fun loadPromptJson(): String {
        val path = rerankDspyFilePath
        if (path != null) {
            val file = File(path)
            if (file.exists()) {
                return file.readText()
            }
        }
        return BEST_DSPY_PROMPT_JSON
    }
}
