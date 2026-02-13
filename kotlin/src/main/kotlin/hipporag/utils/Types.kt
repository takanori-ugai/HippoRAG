package hipporag.utils

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class NerRawOutput(
    val chunkId: String,
    val response: String?,
    val uniqueEntities: List<String>,
    val metadata: Map<String, @Contextual Any?>,
)

@Serializable
data class TripleRawOutput(
    val chunkId: String,
    val response: String?,
    val triples: List<List<String>>,
    val metadata: Map<String, @Contextual Any?>,
)

@Serializable
data class QuerySolution(
    val question: String,
    val docs: List<String>,
    var docScores: DoubleArray? = null,
    var answer: String? = null,
    var goldAnswers: MutableList<String>? = null,
    var goldDocs: MutableList<String>? = null,
)

@Serializable
data class LinkingOutput(
    val score: DoubleArray,
    val type: LinkingOutputType,
)

@Serializable
enum class LinkingOutputType {
    NODE,
    DPR,
}

@Serializable
data class EmbeddingRow(
    val hashId: String,
    val content: String,
    val name: String? = null,
) {
    fun toAttributes(): Map<String, Any> =
        mapOf(
            "hash_id" to hashId,
            "content" to content,
            "name" to (name ?: hashId),
        )
}

@Serializable
data class Message(
    val role: String,
    val content: String,
)

@Serializable
data class LlmResult(
    val response: String,
    val metadata: Map<String, @Contextual Any?>,
    val cacheHit: Boolean = false,
)

@Serializable
data class RagQaResult(
    val solutions: List<QuerySolution>,
    val responseMessages: List<String>,
    val metadata: List<Map<String, @Contextual Any?>>,
    val overallRetrievalResult: Map<String, Double>?,
    val overallQaResults: Map<String, Double>?,
)

@Serializable
data class OpenieResults(
    val docs: List<OpenieDoc>,
    @SerialName("avg_ent_chars") val avgEntChars: Double,
    @SerialName("avg_ent_words") val avgEntWords: Double,
)

@Serializable
data class OpenieDoc(
    val idx: String,
    val passage: String,
    @SerialName("extracted_entities") val extractedEntities: List<String>,
    @SerialName("extracted_triples") val extractedTriples: List<List<String>>,
)
