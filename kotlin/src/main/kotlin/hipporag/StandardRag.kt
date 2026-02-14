package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.QuerySolution
import hipporag.utils.RagQaResult

/**
 * Lightweight DPR-only RAG wrapper over [HippoRag].
 *
 * This variant skips graph-based retrieval and relies on dense passage retrieval.
 */
class StandardRag(
    initialConfig: BaseConfig? = null,
    saveDir: String? = null,
    llmModelName: String? = null,
    embeddingModelName: String? = null,
    llmBaseUrl: String? = null,
    embeddingBaseUrl: String? = null,
    azureEndpoint: String? = null,
    azureEmbeddingEndpoint: String? = null,
) {
    private val config: BaseConfig = (initialConfig ?: BaseConfig()).copy()
    private val hippoRag: HippoRag

    init {
        if (saveDir != null) config.saveDir = saveDir
        if (llmModelName != null) config.llmName = llmModelName
        if (embeddingModelName != null) config.embeddingModelName = embeddingModelName
        if (llmBaseUrl != null) config.llmBaseUrl = llmBaseUrl
        if (embeddingBaseUrl != null) config.embeddingBaseUrl = embeddingBaseUrl
        if (azureEndpoint != null) config.azureEndpoint = azureEndpoint
        if (azureEmbeddingEndpoint != null) config.azureEmbeddingEndpoint = azureEmbeddingEndpoint

        hippoRag = HippoRag(config = config)
    }

    /** Indexes documents for DPR-only retrieval. */
    fun index(docs: List<String>) {
        hippoRag.index(docs)
    }

    /** Deletes documents from the underlying index. */
    fun delete(docsToDelete: List<String>) {
        hippoRag.delete(docsToDelete)
    }

    /**
     * Retrieves passages using DPR only.
     *
     * @return a pair of solutions and optional recall metrics (when [goldDocs] is provided).
     */
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

    /**
     * Runs DPR-only retrieval + QA for [queries].
     */
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

    /**
     * Runs QA over precomputed DPR solutions.
     */
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
