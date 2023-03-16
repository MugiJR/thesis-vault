package thesis.vault;

import com.github.javaparser.ParseProblemException;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.ArrayCreationLevel;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.EnumConstantDeclaration;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.*;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Predicate;
import java.util.logging.Logger;

import static thesis.vault.Validation.*;

public class AppHelper {

    private static final Logger LOGGER = Logger.getLogger(
            Thread.currentThread().getStackTrace()[0].getClassName() );


    // file extension type
    private String fileType;

    // Archive path to keep all parsed json files
    private String archivePath;

    private StringBuilder jsonLog;

    AppHelper(String fileType){
        this.fileType = fileType;
        this.jsonLog = new StringBuilder();
        this.archivePath = createJsonArchiveFolder();
    }

    Predicate<String> filterOutUnwantedDirectories = (directory) -> directory.startsWith(".")
            || directory.equals("bin")
            || directory.equals("doc");


    public void retrieveAllFiles(File[] arr, int index, int level, StringBuilder log) {
        // terminate condition
        if (index == arr.length)
            return;

        // tabs for internal levels
        for (int i = 0; i < level; i++)
            if ((arr[index].isFile() && arr[index].getName().endsWith(fileType)) || arr[index].isDirectory()){
                log.append("\t");
            }


        // for files
        if (arr[index].isFile() && arr[index].getName().endsWith(fileType)) {
            log.append(arr[index].getName()).append(" ----- ");
            try {
                CompilationUnit cu = StaticJavaParser.parse(new File(arr[index].getAbsolutePath()));
                LOGGER.fine("Parsing "+ arr[index].getName());
                toJson(cu.getChildNodes());
                String fileNameWithoutExt = arr[index].getName().substring(0,arr[index].getName().lastIndexOf("."));
                String jsonArchivesPath = archivePath+"\\"+fileNameWithoutExt+".json";
                PrintWriter pw = new PrintWriter(jsonArchivesPath);
                pw.print(jsonLog);
                pw.flush();
                pw.close();
                if (isStringValidJson(jsonLog.toString())) {
                    log.append("PARSED");
                    LOGGER.fine("Generated "+ fileNameWithoutExt + ".json successfully ");
                }
                else {
                    log.append("JSON ERROR");
                    LOGGER.severe("Error while generating "+ fileNameWithoutExt + ".json");
                }
            } catch (ParseProblemException e) {
                log.append("NON COMPILABLE");
                LOGGER.info("Ignored " + arr[index].getName() + " due to parsing issue with the file.");
            } catch (Exception e) {
                log.append("Unknown Exception");
                LOGGER.severe("Exception in " + arr[index].getName() + " " + e);
            } finally {
                // Reset the jsonLog for the next file
                jsonLog = new StringBuilder();
            }
            log.append("\n");
        }

        // for sub-directories
        else if (arr[index].isDirectory() && !filterOutUnwantedDirectories.test(arr[index].getName())) {
            log.append("[").append(arr[index].getName()).append("]").append("\n");
            // recursion for sub-directories
            retrieveAllFiles(arr[index].listFiles(), 0, level + 1, log);
        }

        // recursion for main directory
        retrieveAllFiles(arr, ++index, level, log);
    }

    private void toJson(List<Node> nodes) {
        jsonLog.append("[");
        for (int idx = 0; idx < nodes.size(); idx++) {
            Node node = nodes.get(idx);
            if (node instanceof UnparsableStmt /*|| node instanceof EmptyStmt*/) {
                throw new IllegalArgumentException(node.getClass() + " " + node);
            }
            if (node instanceof Name ||
                    node instanceof SimpleName ||
                    node instanceof AnnotationExpr ||
                    node instanceof ClassExpr ||
                    node instanceof EnclosedExpr ||
                    node instanceof SuperExpr ||
                    node instanceof ThisExpr ||
                    node instanceof TypeExpr ||
                    node instanceof VoidType ||
                    node instanceof Modifier ||
                    node instanceof LiteralExpr ||
                    node instanceof ContinueStmt ||
                    node instanceof LineComment ||
                    node instanceof BlockComment ||
                    node instanceof PrimitiveType ||
                    node instanceof ArrayInitializerExpr ||
                    node instanceof ArrayCreationLevel ||
                    node instanceof MarkerAnnotationExpr ||
                    node instanceof SingleMemberAnnotationExpr ||
                    node instanceof NormalAnnotationExpr ||
                    node instanceof UnknownType ||
                    node instanceof VarType ||
                    node instanceof WildcardType ||
                    node instanceof BlockStmt ||
                    node instanceof BreakStmt ||
                    node instanceof EmptyStmt ||
                    node instanceof ReturnStmt ||
                    node instanceof SwitchEntry ||
                    node instanceof ExplicitConstructorInvocationStmt ||
                    node instanceof EnumConstantDeclaration ||
                    node instanceof JavadocComment
            ) {
                jsonLog.append("{\"type\":\"")
                        .append(node.getClass().getName())
                        .append("\", \"data\":\"")
                        .append(node.toString().replace("\\", "\\\\")
                                .replace("\n", "\\n")
                                .replace("\r", "\\r")
                                .replace("\"", "\\\""));

                if (node instanceof Name ||
                        node instanceof BlockStmt ||
                        node instanceof BreakStmt ||
                        node instanceof SwitchEntry ||
                        node instanceof ContinueStmt ||
                        node instanceof PrimitiveType ||
                        node instanceof LiteralExpr ||
                        node instanceof SuperExpr ||
                        node instanceof MarkerAnnotationExpr ||
                        node instanceof NormalAnnotationExpr ||
                        node instanceof ReturnStmt ||
                        node instanceof EnumConstantDeclaration ||
                        node instanceof EnclosedExpr ||
                        node instanceof ClassExpr ||
                        node instanceof TypeExpr ||
                        node instanceof WildcardType ||
                        node instanceof ThisExpr ||
                        node instanceof SingleMemberAnnotationExpr ||
                        node instanceof ArrayInitializerExpr ||
                        node instanceof ArrayCreationLevel ||
                        node instanceof ExplicitConstructorInvocationStmt
                ) {
                    jsonLog.append("\", \"child\":");
                    toJson(node.getChildNodes());
                    jsonLog.append("}");
                } else {
                    if (node.getChildNodes().size() != 0) LOGGER.severe("Forgot " + node.getClass().getName());
                    jsonLog.append("\"}");
                }
            } else {
                jsonLog.append("{\"type\":\"")
                        .append(node.getClass().getName())
                        .append("\", \"child\":");
                toJson(node.getChildNodes());
                jsonLog.append("}");
                if (node.getChildNodes().size() == 0)
                    LOGGER.info("Check " + node.getClass().getName() + " " + node);
            }
            if (idx != nodes.size()-1) jsonLog.append(", ");

        }
        jsonLog.append("]");
    }

    private String createJsonArchiveFolder() {
        String currentDir = Paths.get("").toAbsolutePath().toString();
        String jsonArchivePath = currentDir+"\\jsonArchives";
        File file = new File(jsonArchivePath);
        /* Create new directory if not exists */
        if (!file.exists()) file.mkdir();
        return file.getAbsolutePath();
    }

}

