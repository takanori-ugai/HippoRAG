import org.jlleitschuh.gradle.ktlint.reporter.ReporterType
import java.io.File

plugins {
    kotlin("jvm") version "2.4.10"
    application
    id("com.gradleup.shadow") version "9.4.3"
    kotlin("plugin.serialization") version "2.4.10"
    id("org.jlleitschuh.gradle.ktlint") version "14.2.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
}

detekt {
    buildUponDefaultConfig = true
    config.setFrom(files("config/detekt.yml"))
}

group = "com.hipporag"
version = "0.0.1"

application {
    val requestedMain =
        if (project.hasProperty("mainClass")) {
            project.property("mainClass") as String
        } else {
            null
        }
    mainClass.set(requestedMain ?: "hipporag.MainKt")
}

repositories {
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    implementation("ch.qos.logback:logback-classic:1.5.38")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("io.github.oshai:kotlin-logging-jvm:8.0.4")
    implementation("com.github.haifengl:smile-core:4.4.2")

    // LangChain4j dependencies
    implementation("dev.langchain4j:langchain4j:1.18.0")
    implementation("dev.langchain4j:langchain4j-open-ai:1.18.0")
    implementation("dev.langchain4j:langchain4j-azure-open-ai:1.18.0")
    implementation("dev.langchain4j:langchain4j-ollama:1.18.0")
    implementation("dev.langchain4j:langchain4j-community-neo4j:1.18.0-beta27")

    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.4.10")
    testImplementation("io.mockk:mockk:1.14.11")
}

tasks {
    withType<Test> {
        jvmArgs("-XX:+EnableDynamicAgentLoading")
        testLogging {
            events("passed", "skipped", "failed", "standardOut", "standardError")
            showStandardStreams = true
        }
    }

    val execute by registering(JavaExec::class) {
        group = "application"
        mainClass.set(
            if (project.hasProperty("mainClass")) {
                project.property("mainClass") as String
            } else {
                application.mainClass.get()
            },
        )
        classpath = sourceSets.main.get().runtimeClasspath
    }
}

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}
