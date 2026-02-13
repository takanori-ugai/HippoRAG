package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.QuerySolution
import hipporag.utils.RagQaResult

// Lightweight Kotlin port of ../src/hipporag/StandardRAG.py using DPR-only retrieval.
class StandardRag(
    globalConfig: BaseConfig? = null,
    saveDir: String? = null,
    llmModelName: String? = null,
    embeddingModelName: String? = null,
    llmBaseUrl: String? = null,
    azureEndpoint: String? = null,
    azureEmbeddingEndpoint: String? = null,
) {
    private val config: BaseConfig = globalConfig ?: BaseConfig()
    private val hippoRag: HippoRag

    init {
        if (saveDir != null) config.saveDir = saveDir
        if (llmModelName != null) config.llmName = llmModelName
        if (embeddingModelName != null) config.embeddingModelName = embeddingModelName
        if (llmBaseUrl != null) config.llmBaseUrl = llmBaseUrl
        if (azureEndpoint != null) config.azureEndpoint = azureEndpoint
        if (azureEmbeddingEndpoint != null) config.azureEmbeddingEndpoint = azureEmbeddingEndpoint

        hippoRag = HippoRag(globalConfig = config)
    }

    fun index(docs: List<String>) {
        hippoRag.index(docs)
    }

    fun delete(docsToDelete: List<String>) {
        hippoRag.delete(docsToDelete)
    }

    fun retrieve(
        queries: List<String>,
        numToRetrieve: Int? = null,
        goldDocs: List<List<String>>? = null,
    ): Pair<List<QuerySolution>, Map<String, Double>?> =
        hippoRag.retrieveDpr(
            queries = queries,
            numToRetrieve = numToRetrieve,
            goldDocs = goldDocs,
        )

    fun ragQa(
        queries: List<String>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult =
        hippoRag.ragQaDpr(
            queries = queries,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
        )

    fun ragQaWithSolutions(
        queries: List<QuerySolution>,
        goldDocs: List<List<String>>? = null,
        goldAnswers: List<List<String>>? = null,
    ): RagQaResult =
        hippoRag.ragQaDprWithSolutions(
            queries = queries,
            goldDocs = goldDocs,
            goldAnswers = goldAnswers,
        )
}
