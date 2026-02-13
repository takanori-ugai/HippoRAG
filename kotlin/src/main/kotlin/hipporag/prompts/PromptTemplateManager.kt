@file:OptIn(ExperimentalSerializationApi::class)

package hipporag.prompts

import hipporag.utils.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonClassDiscriminator
import java.io.File
import kotlin.OptIn

class PromptTemplateManager(
    private val roleMapping: Map<String, String>,
) {
    private val logger = KotlinLogging.logger {}
    private val templates: Map<String, PromptTemplate> = PromptTemplates.templates

    fun render(
        name: String,
        promptUser: String,
    ): List<Message> = render(name, mapOf("prompt_user" to promptUser))

    fun render(
        name: String,
        variables: Map<String, String>,
    ): List<Message> {
        val template = templates[name] ?: throw IllegalArgumentException("Template '$name' not found.")
        return when (template) {
            is PromptTemplate.Text -> {
                listOf(
                    Message(role = roleMapping["user"] ?: "user", content = substitute(template.template, variables)),
                )
            }

            is PromptTemplate.Chat -> {
                template.messages.map { msg ->
                    val role = roleMapping[msg.role] ?: msg.role
                    Message(role = role, content = substitute(msg.content, variables))
                }
            }
        }
    }

    fun listTemplateNames(): List<String> = templates.keys.toList()

    fun isTemplateNameValid(name: String): Boolean = templates.containsKey(name)

    private fun substitute(
        template: String,
        variables: Map<String, String>,
    ): String {
        val pattern = Regex("""\$\{(\w+)\}""")
        return pattern.replace(template) { match ->
            val key = match.groupValues[1]
            variables[key] ?: run {
                logger.warn { "Missing prompt variable '$key'." }
                match.value
            }
        }
    }
}

@Serializable
@JsonClassDiscriminator("type")
sealed class PromptTemplate {
    @Serializable
    @SerialName("text")
    data class Text(
        val template: String,
    ) : PromptTemplate()

    @Serializable
    @SerialName("chat")
    data class Chat(
        val messages: List<PromptMessageTemplate>,
    ) : PromptTemplate()
}

@Serializable
data class PromptMessageTemplate(
    val role: String,
    val content: String,
)

object PromptTemplates {
    private val logger = KotlinLogging.logger {}

    val templates: Map<String, PromptTemplate> by lazy {
        loadTemplates()
    }

    private val json =
        Json {
            isLenient = true
            ignoreUnknownKeys = true
        }

    private fun loadTemplates(): Map<String, PromptTemplate> {
        val templatesDir = getResourceAsFile("prompts/templates")
        if (templatesDir == null || !templatesDir.isDirectory) {
            logger.warn { "Templates directory 'prompts/templates' not found in resources." }
            return emptyMap()
        }

        return templatesDir
            .listFiles { _, name -> name.endsWith(".json") }
            ?.mapNotNull { file ->
                val templateName = file.nameWithoutExtension
                try {
                    val content = file.readText()
                    val template = json.decodeFromString<PromptTemplate>(content)
                    templateName to template
                } catch (e: Exception) {
                    logger.error(e) { "Error loading template: ${file.name}" }
                    null
                }
            }?.toMap() ?: emptyMap()
    }

    private fun getResourceAsFile(path: String): File? =
        try {
            this::class.java.classLoader
                .getResource(path)
                ?.let { File(it.toURI()) }
        } catch (e: Exception) {
            logger.error(e) { "Error getting resource: $path" }
            null
        }
}
