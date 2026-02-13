package hipporag

import hipporag.config.BaseConfig
import hipporag.utils.loadConfigFromJson
import hipporag.utils.stringToBool
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    val parsed = Args.parse(args)

    val json = Json { ignoreUnknownKeys = true }
    val docs = json.decodeFromString(ListSerializer(String.serializer()), File(parsed.docsPath).readText())
    val queries = json.decodeFromString(ListSerializer(String.serializer()), File(parsed.queriesPath).readText())

    val config = parsed.configPath?.let { loadConfigFromJson(it) } ?: BaseConfig()
    parsed.saveDir?.let { config.saveDir = it }
    parsed.llmName?.let { config.llmName = it }
    parsed.llmBaseUrl?.let { config.llmBaseUrl = it }
    parsed.embeddingName?.let { config.embeddingModelName = it }
    parsed.openieMode?.let { config.openieMode = it }
    parsed.forceIndexFromScratch?.let { config.forceIndexFromScratch = stringToBool(it) }
    parsed.forceOpenieFromScratch?.let { config.forceOpenieFromScratch = stringToBool(it) }
    parsed.rerankDspyFilePath?.let { config.rerankDspyFilePath = it }

    val hipporag = HippoRag(globalConfig = config)
    hipporag.index(docs)
    hipporag.ragQa(queries = queries, goldDocs = null, goldAnswers = null)
}

private data class Args(
    val docsPath: String,
    val queriesPath: String,
    val configPath: String?,
    val saveDir: String?,
    val llmBaseUrl: String?,
    val llmName: String?,
    val embeddingName: String?,
    val openieMode: String?,
    val forceIndexFromScratch: String?,
    val forceOpenieFromScratch: String?,
    val rerankDspyFilePath: String?,
) {
    companion object {
        fun parse(args: Array<String>): Args {
            val map = mutableMapOf<String, String>()
            var i = 0
            while (i < args.size) {
                val key = args[i]
                if (!key.startsWith("--") || i + 1 >= args.size) {
                    usageAndExit()
                }
                map[key.removePrefix("--")] = args[i + 1]
                i += 2
            }

            val docs = map["docs"] ?: usageAndExit()
            val queries = map["queries"] ?: usageAndExit()

            return Args(
                docsPath = docs,
                queriesPath = queries,
                configPath = map["config"],
                saveDir = map["save_dir"],
                llmBaseUrl = map["llm_base_url"],
                llmName = map["llm_name"],
                embeddingName = map["embedding_name"],
                openieMode = map["openie_mode"],
                forceIndexFromScratch = map["force_index_from_scratch"],
                forceOpenieFromScratch = map["force_openie_from_scratch"],
                rerankDspyFilePath = map["rerank_dspy_file_path"],
            )
        }

        private fun usageAndExit(): Nothing {
            System.err.println(
                "Usage: --docs <docs.json> --queries <queries.json> [--config <config.json>] [--save_dir outputs] " +
                    "[--llm_name gpt-4o-mini] [--llm_base_url <url>] [--embedding_name <name>] " +
                    "[--openie_mode online] [--force_index_from_scratch false] [--force_openie_from_scratch false] " +
                    "[--rerank_dspy_file_path <path>]",
            )
            exitProcess(2)
        }
    }
}
