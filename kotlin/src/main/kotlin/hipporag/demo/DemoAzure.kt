package hipporag.demo

fun main(args: Array<String>) {
    runDemo(args) { config, parsed ->
        config.azureEndpoint = parsed.azureEndpoint
        config.azureEmbeddingEndpoint = parsed.azureEmbeddingEndpoint
    }
}
