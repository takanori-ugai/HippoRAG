package hipporag.llm

import dev.langchain4j.model.azure.AzureOpenAiChatModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.openai.OpenAiChatModel
import hipporag.config.BaseConfig
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

fun getLlm(globalConfig: BaseConfig): BaseLLM {
    val provider = globalConfig.llmProvider?.lowercase()
    val modelName = globalConfig.llmName
    val temperature = globalConfig.temperature

    val unsupportedProviders =
        setOf(
            "bedrock",
            "bedrock_llm",
            "aws_bedrock",
            "transformers",
            "transformers_llm",
            "transformers_offline",
            "transformers-offline",
            "vllm",
            "vllm_offline",
            "vllm-offline",
        )

    if (provider in unsupportedProviders) {
        error(
            "LLM provider '$provider' is not supported in the Kotlin port yet. " +
                "Supported providers: OpenAI-compatible (default), Azure (set azureEndpoint), " +
                "and Ollama (set llmProvider=ollama or use an Ollama base URL).",
        )
    } else if (provider != null && provider !in setOf("openai", "azure", "ollama")) {
        logger.warn { "Unknown LLM provider '$provider'. Falling back to OpenAI-compatible settings." }
    }

    val model =
        when {
            globalConfig.azureEndpoint != null -> {
                val apiKey = globalConfig.azureApiKey ?: System.getenv("AZURE_OPENAI_API_KEY")
                val deployment = globalConfig.azureDeploymentName ?: modelName
                AzureOpenAiChatModel
                    .builder()
                    .apiKey(apiKey)
                    .endpoint(globalConfig.azureEndpoint)
                    .deploymentName(deployment)
                    .temperature(temperature)
                    .build()
            }

            provider == "ollama" || (globalConfig.llmBaseUrl?.contains("ollama") == true) -> {
                val baseUrl = globalConfig.ollamaBaseUrl ?: globalConfig.llmBaseUrl ?: "http://localhost:11434"
                val ollamaModel = globalConfig.ollamaModelName ?: modelName
                OllamaChatModel
                    .builder()
                    .baseUrl(baseUrl)
                    .modelName(ollamaModel)
                    .temperature(temperature)
                    .build()
            }

            else -> {
                val apiKey = globalConfig.openAiApiKey ?: System.getenv("OPENAI_API_KEY")
                val builder =
                    OpenAiChatModel
                        .builder()
                        .apiKey(apiKey)
                        .modelName(modelName)
                        .temperature(temperature)
                if (globalConfig.llmBaseUrl != null) {
                    builder.baseUrl(globalConfig.llmBaseUrl)
                }
                if (globalConfig.maxNewTokens != null) {
                    builder.maxTokens(globalConfig.maxNewTokens)
                }
                builder.build()
            }
        }

    return LangChainChatLLM(model)
}
