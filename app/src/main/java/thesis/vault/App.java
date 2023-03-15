/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package thesis.vault;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.StaticJavaParser;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;

import com.github.javaparser.ast.body.EnumConstantDeclaration;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.UnknownType;
import com.github.javaparser.ast.type.VarType;
import com.github.javaparser.ast.type.VoidType;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.type.PrimitiveType;

public class App {
    public static void toJson(List<Node> nodes, StringBuilder sb)
    {
        sb.append("[");
        for (Node node : nodes) {
            if (node instanceof UnparsableStmt /*|| node instanceof EmptyStmt*/) {
                throw new IllegalArgumentException(node.getClass() + " " + node);
            }
            if (node instanceof Name ||
                node instanceof SimpleName ||
                node instanceof VoidType ||
                node instanceof Modifier ||
                node instanceof LiteralExpr ||
                node instanceof ContinueStmt ||
                node instanceof LineComment ||
                node instanceof BlockComment ||
                node instanceof PrimitiveType ||
                node instanceof AnnotationExpr ||
                node instanceof ClassExpr ||
                node instanceof EnclosedExpr ||
                node instanceof SuperExpr ||
                node instanceof ThisExpr ||
                node instanceof TypeExpr ||
                node instanceof UnknownType ||
                node instanceof VarType ||
                node instanceof BreakStmt ||
                node instanceof EmptyStmt ||
                node instanceof ReturnStmt ||
                node instanceof EnumConstantDeclaration
            ) {
                sb.append("{\"type\":\"")
                    .append(node.getClass().getName())
                    .append("\", \"data\":\"")
                    .append(node.toString().replace("\\", "\\\\")
                                .replace("\n", "\\n")
                                .replace("\r", "\\r")
                                .replace("\"", "\\\""));

                    if (node instanceof Name ||
                        node instanceof ContinueStmt ||
                        node instanceof PrimitiveType ||
                        node instanceof LiteralExpr ||
                        node instanceof ReturnStmt ||
                        node instanceof EnumConstantDeclaration
                    ) {

                        sb.append("\", \"child\":");
                        toJson(node.getChildNodes(), sb);
                        sb.append("}");
                    } else {
                        if (node.getChildNodes().size() != 0) System.out.println("Forgot " + node.getClass().getName());
                        sb.append("\"}");
                    }
            } else {
                sb.append("{\"type\":\"")
                    .append(node.getClass().getName())
                    .append("\", \"child\":");
                toJson(node.getChildNodes(), sb);
                sb.append("}");
                if (node.getChildNodes().size() == 0) System.out.println("Check " + node.getClass().getName() + " " + node);
            }            
            if (!node.equals(nodes.get(nodes.size()-1))) sb.append(", ");

        }
        sb.append("]");
    }


    public static void main(String[] args) {
        System.out.println("ARGS Path : " + args[0]);
        try {
            StringBuilder sb = new StringBuilder();
            //"*.java"
            CompilationUnit cu = StaticJavaParser.parse(new File(args[0]));
            toJson(cu.getChildNodes(), sb);
            PrintWriter pw = new PrintWriter(new File("result.json"));
            pw.print(sb);
            System.out.println(sb);
            pw.close();
        } catch (FileNotFoundException e) {
            System.out.println(e);
        }
    }
}
