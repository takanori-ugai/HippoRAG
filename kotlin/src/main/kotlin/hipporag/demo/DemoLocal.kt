package hipporag.demo

fun main(args: Array<String>) {
    runDemo(args) { config, parsed ->
        config.llmProvider = "ollama"
        config.ollamaBaseUrl = parsed.llmBaseUrl ?: "http://localhost:11434"
        config.embeddingProvider = "ollama"
        config.embeddingBaseUrl = parsed.embeddingBaseUrl ?: "http://localhost:11434"
    }
}
