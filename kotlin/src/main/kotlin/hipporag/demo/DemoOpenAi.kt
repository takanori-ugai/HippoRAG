package hipporag.demo

import hipporag.utils.stringToBool

fun main(args: Array<String>) {
    runDemo(args) { config, parsed ->
        config.llmBaseUrl = parsed.llmBaseUrl
        config.embeddingBaseUrl = parsed.embeddingBaseUrl
        config.forceIndexFromScratch = stringToBool("false")
        config.forceOpenieFromScratch = stringToBool("false")
    }
}
