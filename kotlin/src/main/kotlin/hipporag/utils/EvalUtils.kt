package hipporag.utils

private const val PUNCTUATION_CHARS = "!\"#$%&'()*+,-./:;<>?@[\\]^_`{|}~"

fun normalizeAnswer(answer: String): String {
    fun removeArticles(text: String): String = Regex("\\b(a|an|the)\\b").replace(text, " ")

    fun whiteSpaceFix(text: String): String =
        text
            .trim()
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .joinToString(" ")

    fun removePunc(text: String): String = text.filter { ch -> ch !in PUNCTUATION_CHARS }

    fun lower(text: String): String = text.lowercase()

    return whiteSpaceFix(removeArticles(removePunc(lower(answer))))
}
