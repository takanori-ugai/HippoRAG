@file:Suppress("TooGenericExceptionCaught")

package hipporag.informationextraction

import hipporag.llm.BaseLLM
import hipporag.prompts.PromptTemplateManager
import hipporag.utils.EmbeddingRow
import hipporag.utils.NerRawOutput
import hipporag.utils.TripleRawOutput
import hipporag.utils.extractJsonObjectWithKey
import hipporag.utils.filterInvalidTriples
import hipporag.utils.jsonWithDefaults
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

interface OpenIEBase {
    fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>>
}

class OpenIE(
    private val llmModel: BaseLLM,
) : OpenIEBase {
    private val logger = KotlinLogging.logger {}
    private val promptTemplateManager =
        PromptTemplateManager(
            roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
        )
    private val json = jsonWithDefaults { ignoreUnknownKeys = true }

    private fun ner(
        chunkKey: String,
        passage: String,
    ): NerRawOutput {
        val messages = promptTemplateManager.render("ner", mapOf("passage" to passage))
        return try {
            val result = llmModel.infer(messages)
            val entities = extractNamedEntitiesFromResponse(result.response, json)
            NerRawOutput(
                chunkId = chunkKey,
                response = result.response,
                uniqueEntities = entities.distinct(),
                metadata = result.metadata,
            )
        } catch (e: Exception) {
            logger.warn(e) { "NER failed for chunk $chunkKey" }
            NerRawOutput(
                chunkId = chunkKey,
                response = null,
                uniqueEntities = emptyList(),
                metadata = mapOf("error" to e.message.orEmpty()),
            )
        }
    }

    private fun tripleExtraction(
        chunkKey: String,
        passage: String,
        namedEntities: List<String>,
    ): TripleRawOutput {
        val namedEntityJson = buildNamedEntityJson(namedEntities, json)
        val messages =
            promptTemplateManager.render(
                "triple_extraction",
                mapOf("passage" to passage, "named_entity_json" to namedEntityJson),
            )
        return try {
            val result = llmModel.infer(messages)
            val triples = extractTriplesFromResponse(result.response, json)
            TripleRawOutput(
                chunkId = chunkKey,
                response = result.response,
                triples = filterInvalidTriples(triples),
                metadata = result.metadata,
            )
        } catch (e: Exception) {
            logger.warn(e) { "Triple extraction failed for chunk $chunkKey" }
            TripleRawOutput(
                chunkId = chunkKey,
                response = null,
                triples = emptyList(),
                metadata = mapOf("error" to e.message.orEmpty()),
            )
        }
    }

    override fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>> {
        val nerResults = mutableMapOf<String, NerRawOutput>()
        val tripleResults = mutableMapOf<String, TripleRawOutput>()

        for ((chunkKey, row) in rows) {
            val nerOutput = ner(chunkKey, row.content)
            nerResults[chunkKey] = nerOutput
            val tripleOutput = tripleExtraction(chunkKey, row.content, nerOutput.uniqueEntities)
            tripleResults[chunkKey] = tripleOutput
        }

        return nerResults to tripleResults
    }
}

class VllmOfflineOpenIE(
    private val llmModel: BaseLLM,
) : OpenIEBase {
    private val logger = KotlinLogging.logger {}
    private val promptTemplateManager =
        PromptTemplateManager(
            roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
        )
    private val json = jsonWithDefaults { ignoreUnknownKeys = true }

    override fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>> {
        val nerResults = mutableMapOf<String, NerRawOutput>()
        val tripleResults = mutableMapOf<String, TripleRawOutput>()

        val namedEntitiesByChunk = mutableMapOf<String, List<String>>()
        for ((chunkKey, row) in rows) {
            val messages = promptTemplateManager.render("ner", mapOf("passage" to row.content))
            try {
                val result = llmModel.infer(messages)
                val entities = extractNamedEntitiesFromResponse(result.response, json)
                if (entities.isEmpty()) {
                    logger.warn { "No entities extracted for chunk_id: $chunkKey" }
                }
                nerResults[chunkKey] =
                    NerRawOutput(
                        chunkId = chunkKey,
                        response = result.response,
                        uniqueEntities = entities.distinct(),
                        metadata = result.metadata,
                    )
                namedEntitiesByChunk[chunkKey] = entities
            } catch (e: Exception) {
                logger.warn(e) { "NER failed for chunk $chunkKey" }
                nerResults[chunkKey] =
                    NerRawOutput(
                        chunkId = chunkKey,
                        response = null,
                        uniqueEntities = emptyList(),
                        metadata = mapOf("error" to e.message.orEmpty()),
                    )
                namedEntitiesByChunk[chunkKey] = emptyList()
            }
        }

        for ((chunkKey, row) in rows) {
            val namedEntities = namedEntitiesByChunk[chunkKey] ?: emptyList()
            val namedEntityJson = buildNamedEntityJson(namedEntities, json)
            val messages =
                promptTemplateManager.render(
                    "triple_extraction",
                    mapOf("passage" to row.content, "named_entity_json" to namedEntityJson),
                )
            try {
                val result = llmModel.infer(messages)
                val triples = extractTriplesFromResponse(result.response, json)
                if (triples.isEmpty()) {
                    logger.warn { "No triples extracted for chunk_id: $chunkKey" }
                }
                tripleResults[chunkKey] =
                    TripleRawOutput(
                        chunkId = chunkKey,
                        response = result.response,
                        triples = filterInvalidTriples(triples),
                        metadata = result.metadata,
                    )
            } catch (e: Exception) {
                logger.warn(e) { "Triple extraction failed for chunk $chunkKey" }
                tripleResults[chunkKey] =
                    TripleRawOutput(
                        chunkId = chunkKey,
                        response = null,
                        triples = emptyList(),
                        metadata = mapOf("error" to e.message.orEmpty()),
                    )
            }
        }

        return nerResults to tripleResults
    }
}

class TransformersOfflineOpenIE(
    private val llmModel: BaseLLM,
) : OpenIEBase {
    private val logger = KotlinLogging.logger {}
    private val promptTemplateManager =
        PromptTemplateManager(
            roleMapping = mapOf("system" to "system", "user" to "user", "assistant" to "assistant"),
        )
    private val json = jsonWithDefaults { ignoreUnknownKeys = true }

    override fun batchOpenie(rows: Map<String, EmbeddingRow>): Pair<Map<String, NerRawOutput>, Map<String, TripleRawOutput>> {
        val nerResults = mutableMapOf<String, NerRawOutput>()
        val tripleResults = mutableMapOf<String, TripleRawOutput>()

        val namedEntitiesByChunk = mutableMapOf<String, List<String>>()
        for ((chunkKey, row) in rows) {
            val messages = promptTemplateManager.render("ner", mapOf("passage" to row.content))
            try {
                val result = llmModel.infer(messages)
                val entities = extractNamedEntitiesFromResponse(result.response, json)
                if (entities.isEmpty()) {
                    logger.warn { "No entities extracted for chunk_id: $chunkKey" }
                }
                nerResults[chunkKey] =
                    NerRawOutput(
                        chunkId = chunkKey,
                        response = result.response,
                        uniqueEntities = entities.distinct(),
                        metadata = result.metadata,
                    )
                namedEntitiesByChunk[chunkKey] = entities
            } catch (e: Exception) {
                logger.warn(e) { "NER failed for chunk $chunkKey" }
                nerResults[chunkKey] =
                    NerRawOutput(
                        chunkId = chunkKey,
                        response = null,
                        uniqueEntities = emptyList(),
                        metadata = mapOf("error" to e.message.orEmpty()),
                    )
                namedEntitiesByChunk[chunkKey] = emptyList()
            }
        }

        for ((chunkKey, row) in rows) {
            val namedEntities = namedEntitiesByChunk[chunkKey] ?: emptyList()
            val namedEntityJson = buildNamedEntityJson(namedEntities, json)
            val messages =
                promptTemplateManager.render(
                    "triple_extraction",
                    mapOf("passage" to row.content, "named_entity_json" to namedEntityJson),
                )
            try {
                val result = llmModel.infer(messages)
                val triples = extractTriplesFromResponse(result.response, json)
                if (triples.isEmpty()) {
                    logger.warn { "No triples extracted for chunk_id: $chunkKey" }
                }
                tripleResults[chunkKey] =
                    TripleRawOutput(
                        chunkId = chunkKey,
                        response = result.response,
                        triples = filterInvalidTriples(triples),
                        metadata = result.metadata,
                    )
            } catch (e: Exception) {
                logger.warn(e) { "Triple extraction failed for chunk $chunkKey" }
                tripleResults[chunkKey] =
                    TripleRawOutput(
                        chunkId = chunkKey,
                        response = null,
                        triples = emptyList(),
                        metadata = mapOf("error" to e.message.orEmpty()),
                    )
            }
        }

        return nerResults to tripleResults
    }
}

private fun buildNamedEntityJson(
    namedEntities: List<String>,
    json: Json,
): String =
    json.encodeToString(
        JsonObject.serializer(),
        JsonObject(mapOf("named_entities" to JsonArray(namedEntities.map { JsonPrimitive(it) }))),
    )

private fun extractNamedEntitiesFromResponse(
    response: String,
    json: Json,
): List<String> {
    val jsonObject = extractJsonObjectWithKey(response, "named_entities", json) ?: return emptyList()
    val entities = jsonObject["named_entities"]?.jsonArray ?: return emptyList()
    return entities.mapNotNull { it.jsonPrimitive.contentOrNull?.trim() }.filter { it.isNotEmpty() }
}

private fun extractTriplesFromResponse(
    response: String,
    json: Json,
): List<List<String>> {
    val jsonObject = extractJsonObjectWithKey(response, "triples", json) ?: return emptyList()
    val triplesArray = jsonObject["triples"]?.jsonArray ?: return emptyList()
    val triples = mutableListOf<List<String>>()
    for (tripleEl in triplesArray) {
        val tripleArray = tripleEl.asJsonArrayOrNull() ?: continue
        val triple = tripleArray.mapNotNull { it.jsonPrimitive.contentOrNull?.trim() }.filter { it.isNotEmpty() }
        if (triple.isNotEmpty()) {
            triples.add(triple)
        }
    }
    return triples
}

private fun JsonElement.asJsonArrayOrNull(): JsonArray? = runCatching { this.jsonArray }.getOrNull()
