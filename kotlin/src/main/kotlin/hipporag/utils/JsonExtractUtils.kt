package hipporag.utils

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

fun extractJsonObjectWithKey(
    response: String,
    key: String,
    json: Json,
): JsonObject? {
    val escapedKey = Regex.escape(key)
    val pattern =
        Regex(
            "\\{[^{}]*\"$escapedKey\"\\s*:\\s*\\[[\\s\\S]*?\\][^{}]*\\}",
            RegexOption.DOT_MATCHES_ALL,
        )
    val match = pattern.find(response) ?: return null
    return runCatching { json.parseToJsonElement(match.value).jsonObject }.getOrNull()
}
