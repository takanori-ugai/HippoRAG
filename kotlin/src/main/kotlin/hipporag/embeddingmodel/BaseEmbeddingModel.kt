package hipporag.embeddingmodel

import hipporag.config.BaseConfig

interface BaseEmbeddingModel {
    fun batchEncode(
        texts: List<String>,
        instruction: String? = null,
        norm: Boolean = false,
    ): Array<DoubleArray>
}

interface EmbeddingModelFactory {
    fun create(
        globalConfig: BaseConfig,
        embeddingModelName: String,
    ): BaseEmbeddingModel
}

fun getEmbeddingModel(name: String): EmbeddingModelFactory = LangChainEmbeddingFactory(name)
