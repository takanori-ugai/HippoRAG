package hipporag.demo

import hipporag.HippoRag
import hipporag.config.BaseConfig
import hipporag.utils.stringToBool
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import java.io.File

fun main(args: Array<String>) {
    val parsed = DemoArgs.parse(args)
    val json = Json { ignoreUnknownKeys = true }
    val docs = json.decodeFromString(ListSerializer(String.serializer()), File(parsed.docsPath).readText())
    val queries = json.decodeFromString(ListSerializer(String.serializer()), File(parsed.queriesPath).readText())

    val config =
        BaseConfig().apply {
            saveDir = parsed.saveDir
            llmName = parsed.llmName
            llmBaseUrl = parsed.llmBaseUrl
            embeddingModelName = parsed.embeddingName
            embeddingBaseUrl = parsed.embeddingBaseUrl
            openieMode = "online"
            forceIndexFromScratch = stringToBool("false")
            forceOpenieFromScratch = stringToBool("false")
        }

    val hipporag = HippoRag(globalConfig = config)
    hipporag.index(docs)
    hipporag.ragQa(queries = queries, goldDocs = null, goldAnswers = null)
}
