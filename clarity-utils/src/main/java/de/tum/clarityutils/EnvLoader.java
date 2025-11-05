package de.tum.clarityutils;

import io.github.cdimascio.dotenv.Dotenv;

import java.util.List;

/**
 * Loads environment variables from a .env file using cdimascio's dotenv.
 *
 * <p>The loader attempts to locate a .env file in several candidate directories and falls back
 * to the directory specified by the system property "DOTENV_DIR" if set. If no .env is found,
 * the loader will behave gracefully (ignoreIfMissing).</p>
 */
public class EnvLoader {
    private static final Dotenv dotenv;

    private static final List<String> possibleDirs = List.of(
            "..",
            ".",
            "classpath:.",
            "classpath:..",
            "src/main/resources",
            "src/test/resources"
    );

    static {
        String dir = System.getProperty("DOTENV_DIR", "");
        if (dir.isEmpty()) {
            for (String possibleDir : possibleDirs) {
                try {
                    Dotenv tempDotenv = Dotenv.configure().directory(possibleDir).ignoreIfMissing().load();
                    if (tempDotenv.entries().iterator().hasNext()) {
                        dir = possibleDir;
                        break;
                    }
                } catch (Exception ignored) {
                }
            }
        }
        dotenv = Dotenv.configure()
                       .directory(dir)
                       .ignoreIfMissing()
                       .load();
    }

    /**
     * Return the value of an environment variable loaded from the .env file or system environment.
     *
     * @param key the environment variable name
     * @return the variable value, or null if not present
     */
    public static String get(String key) {
        return dotenv.get(key);
    }
}
