package hipporag.demo

import kotlin.system.exitProcess

data class DemoArgs(
    val docsPath: String,
    val queriesPath: String,
    val saveDir: String,
    val llmName: String,
    val embeddingName: String,
    val llmBaseUrl: String?,
    val embeddingBaseUrl: String?,
    val azureEndpoint: String?,
    val azureEmbeddingEndpoint: String?,
) {
    companion object {
        fun parse(args: Array<String>): DemoArgs {
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

            return DemoArgs(
                docsPath = docs,
                queriesPath = queries,
                saveDir = map["save_dir"] ?: "outputs",
                llmName = map["llm_name"] ?: "gpt-4o-mini",
                embeddingName = map["embedding_name"] ?: "nvidia/NV-Embed-v2",
                llmBaseUrl = map["llm_base_url"],
                embeddingBaseUrl = map["embedding_base_url"],
                azureEndpoint = map["azure_endpoint"],
                azureEmbeddingEndpoint = map["azure_embedding_endpoint"],
            )
        }

        private fun usageAndExit(): Nothing {
            System.err.println(
                "Usage: --docs <docs.json> --queries <queries.json> [--save_dir outputs] " +
                    "[--llm_name gpt-4o-mini] [--llm_base_url <url>] [--embedding_name <name>] " +
                    "[--embedding_base_url <url>] [--azure_endpoint <url>] [--azure_embedding_endpoint <url>]",
            )
            exitProcess(2)
        }
    }
}
