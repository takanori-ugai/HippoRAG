package hipporag.embeddingmodel

import dev.langchain4j.model.azure.AzureOpenAiEmbeddingModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging
import java.net.URI

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
                val apiKey =
                    globalConfig.azureApiKey ?: System.getenv("AZURE_OPENAI_API_KEY")
                        ?: error("Azure API key not found. Set azureApiKey in config or AZURE_OPENAI_API_KEY env var.")
                val deployment = globalConfig.azureEmbeddingDeploymentName ?: embeddingModelName
                AzureOpenAiEmbeddingModel
                    .builder()
                    .apiKey(apiKey)
                    .endpoint(globalConfig.azureEmbeddingEndpoint)
                    .deploymentName(deployment)
                    .build()
            }

            provider == "ollama" -> {
                val baseUrl =
                    globalConfig.ollamaBaseUrl
                        ?: globalConfig.embeddingBaseUrl
                        ?: "http://localhost:11434"
                val model = globalConfig.ollamaEmbeddingModelName ?: embeddingModelName
                OllamaEmbeddingModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(model)
                    .build()
            }

            isLikelyOllamaBaseUrl(globalConfig.embeddingBaseUrl) -> {
                logger.warn {
                    "Ambiguous embedding provider. 'ollama' detected in embeddingBaseUrl, " +
                        "but embeddingProvider is not explicitly 'ollama'. Defaulting to Ollama."
                }
                val baseUrl =
                    globalConfig.ollamaBaseUrl
                        ?: globalConfig.embeddingBaseUrl
                        ?: "http://localhost:11434"
                val model = globalConfig.ollamaEmbeddingModelName ?: embeddingModelName
                OllamaEmbeddingModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(model)
                    .build()
            }

            else -> {
                val apiKey =
                    globalConfig.openAiApiKey ?: System.getenv("OPENAI_API_KEY")
                        ?: error("OpenAI API key not found. Set openAiApiKey in config or OPENAI_API_KEY env var.")
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

    private fun isLikelyOllamaBaseUrl(baseUrl: String?): Boolean {
        if (baseUrl.isNullOrBlank()) return false
        return runCatching {
            val uri = URI(baseUrl)
            uri.port == 11434
        }.getOrDefault(false)
    }
}
