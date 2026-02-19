package hipporag.demo

/**
 * Demo entry point configured for Azure OpenAI endpoints.
 */
fun main(args: Array<String>) {
    runDemo(args) { config, parsed ->
        config.azureEndpoint = parsed.azureEndpoint
        config.azureEmbeddingEndpoint = parsed.azureEmbeddingEndpoint
    }
}
