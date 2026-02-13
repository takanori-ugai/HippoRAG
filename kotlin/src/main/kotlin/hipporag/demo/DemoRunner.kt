package hipporag.demo

import hipporag.HippoRag
import hipporag.config.BaseConfig
import hipporag.utils.jsonWithDefaults
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import java.io.File

fun runDemo(
    args: Array<String>,
    configure: (BaseConfig, DemoArgs) -> Unit,
) {
    val parsed = DemoArgs.parse(args)
    val json = jsonWithDefaults { ignoreUnknownKeys = true }
    val docs = json.decodeFromString(ListSerializer(String.serializer()), File(parsed.docsPath).readText())
    val queries = json.decodeFromString(ListSerializer(String.serializer()), File(parsed.queriesPath).readText())

    val config =
        BaseConfig().apply {
            saveDir = parsed.saveDir
            llmName = parsed.llmName
            embeddingModelName = parsed.embeddingName
            openieMode = "online"
            configure(this, parsed)
        }

    val hipporag = HippoRag(globalConfig = config)
    hipporag.index(docs)
    hipporag.ragQa(queries = queries, goldDocs = null, goldAnswers = null)
}
