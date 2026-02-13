package hipporag.utils

import java.util.regex.Matcher
import java.util.regex.Pattern
import kotlin.math.min
import kotlin.random.Random

fun filterInvalidTriples(triples: List<List<String>>): List<List<String>> {
    val uniqueTriples = mutableSetOf<List<String>>()
    val validTriples = mutableListOf<List<String>>()

    for (triple in triples) {
        if (triple.size != 3) continue
        val validTriple = triple.toList()
        if (uniqueTriples.add(validTriple)) {
            validTriples.add(validTriple)
        }
    }

    return validTriples
}

fun safeUnicodeDecode(content: Any): String {
    val text =
        when (content) {
            is ByteArray -> content.toString(Charsets.UTF_8)
            is String -> content
            else -> throw IllegalArgumentException("Input must be of type ByteArray or String.")
        }
    val unicodeEscapePattern = Regex("\\\\u([0-9a-fA-F]{4})")
    return unicodeEscapePattern.replace(text) { match ->
        val hex = match.groupValues[1]
        hex.toInt(16).toChar().toString()
    }
}

fun fixBrokenGeneratedJson(jsonStr: String): String {
    fun findUnclosed(input: String): List<Char> {
        val unclosed = mutableListOf<Char>()
        var insideString = false
        var escapeNext = false
        for (char in input) {
            if (insideString) {
                when {
                    escapeNext -> escapeNext = false
                    char == '\\' -> escapeNext = true
                    char == '"' -> insideString = false
                }
            } else {
                when (char) {
                    '"' -> {
                        insideString = true
                    }

                    '{', '[' -> {
                        unclosed.add(char)
                    }

                    '}', ']' -> {
                        if (unclosed.isNotEmpty()) {
                            val last = unclosed.last()
                            if ((char == '}' && last == '{') || (char == ']' && last == '[')) {
                                unclosed.removeAt(unclosed.lastIndex)
                            }
                        }
                    }
                }
            }
        }
        return unclosed
    }

    val lastCommaIndex = jsonStr.lastIndexOf(',')
    val truncated = if (lastCommaIndex != -1) jsonStr.substring(0, lastCommaIndex) else jsonStr
    val unclosed = findUnclosed(truncated)
    if (unclosed.isEmpty()) return truncated

    val closingMap = mapOf('{' to '}', '[' to ']')
    val builder = StringBuilder(truncated)
    for (openChar in unclosed.asReversed()) {
        builder.append(closingMap.getValue(openChar))
    }
    return builder.toString()
}

@Suppress("TooGenericExceptionCaught")
fun <T> retryWithBackoff(
    maxAttempts: Int,
    baseDelayMillis: Long = 250,
    maxDelayMillis: Long = 4000,
    jitterMillis: Long = 100,
    retryOn: (Throwable) -> Boolean = { true },
    block: () -> T,
): T {
    require(maxAttempts >= 1) { "maxAttempts must be >= 1" }
    var attempt = 0
    var lastError: Throwable? = null
    while (attempt < maxAttempts) {
        try {
            return block()
        } catch (e: Throwable) {
            lastError = e
            if (!retryOn(e) || attempt == maxAttempts - 1) {
                throw e
            }
            val exponent = 1 shl attempt.coerceAtMost(10)
            val delay = min(maxDelayMillis, baseDelayMillis * exponent.toLong())
            val jitter = if (jitterMillis > 0) Random.nextLong(0, jitterMillis) else 0
            Thread.sleep(delay + jitter)
        }
        attempt += 1
    }
    error("retryWithBackoff: unreachable")
}

fun convertFormatToTemplate(
    originalString: String,
    placeholderMapping: Map<String, String>? = null,
    staticValues: Map<String, Any>? = null,
): String {
    val mapping = placeholderMapping ?: emptyMap()
    val statics = staticValues ?: emptyMap()
    val pattern = Pattern.compile("\\{(\\w+)\\}")
    val matcher = pattern.matcher(originalString)
    val output = StringBuffer()

    while (matcher.find()) {
        val originalPlaceholder = matcher.group(1)
        val replacement =
            when {
                statics.containsKey(originalPlaceholder) -> statics.getValue(originalPlaceholder).toString()
                else -> "\${${mapping[originalPlaceholder] ?: originalPlaceholder}}"
            }
        matcher.appendReplacement(output, Matcher.quoteReplacement(replacement))
    }
    matcher.appendTail(output)
    return output.toString()
}
