package hipporag.embeddingmodel

import dev.langchain4j.model.azure.AzureOpenAiEmbeddingModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging

class LangChainEmbeddingFactory : EmbeddingModelFactory {
    private val logger = KotlinLogging.logger {}

    override fun create(
        globalConfig: BaseConfig,
        embeddingModelName: String,
    ): BaseEmbeddingModel {
        val model = buildEmbeddingModel(globalConfig, embeddingModelName)
        return LangChainEmbeddingModel(model)
    }

    private fun buildEmbeddingModel(
        globalConfig: BaseConfig,
        embeddingModelName: String,
    ): EmbeddingModel {
        val provider = globalConfig.embeddingProvider?.lowercase()
        val unsupportedProviders =
            setOf(
                "cohere",
                "contriever",
                "gritlm",
                "nvembedv2",
                "nv-embed-v2",
                "transformers",
                "vllm",
                "vllm_offline",
                "vllm-offline",
                "huggingface",
            )

        if (provider in unsupportedProviders) {
            error(
                "Embedding provider '$provider' is not supported in the Kotlin port yet. " +
                    "Supported providers: OpenAI-compatible (default), Azure (set azureEmbeddingEndpoint), " +
                    "and Ollama (set embeddingProvider=ollama or use an Ollama base URL).",
            )
        } else if (provider != null && provider !in setOf("openai", "azure", "ollama")) {
            logger.warn { "Unknown embedding provider '$provider'. Falling back to OpenAI-compatible settings." }
        }
        return when {
            globalConfig.azureEmbeddingEndpoint != null -> {
                val apiKey = globalConfig.azureApiKey ?: System.getenv("AZURE_OPENAI_API_KEY")
                val deployment = globalConfig.azureEmbeddingDeploymentName ?: embeddingModelName
                AzureOpenAiEmbeddingModel
                    .builder()
                    .apiKey(apiKey)
                    .endpoint(globalConfig.azureEmbeddingEndpoint)
                    .deploymentName(deployment)
                    .build()
            }

            provider == "ollama" || (globalConfig.embeddingBaseUrl?.contains("ollama") == true) -> {
                val baseUrl = globalConfig.ollamaBaseUrl ?: globalConfig.embeddingBaseUrl ?: "http://localhost:11434"
                val model = globalConfig.ollamaEmbeddingModelName ?: embeddingModelName
                OllamaEmbeddingModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(model)
                    .build()
            }

            else -> {
                val apiKey = globalConfig.openAiApiKey ?: System.getenv("OPENAI_API_KEY")
                val builder =
                    OpenAiEmbeddingModel
                        .builder()
                        .apiKey(apiKey)
                        .modelName(embeddingModelName)
                if (globalConfig.embeddingBaseUrl != null) {
                    builder.baseUrl(globalConfig.embeddingBaseUrl)
                }
                builder.build()
            }
        }
    }
}
