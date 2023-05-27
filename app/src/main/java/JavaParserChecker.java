import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;


public class JavaParserChecker {

    public static Set<String> astTypeSet = new HashSet<>();

    final static Set<String> expectedASTSET = new HashSet<>(
            Arrays.asList("ArrayAccessExpr","ArrayCreationExpr","ArrayInitializerExpr","AssignExpr",
                    "BinaryExpr","BooleanLiteralExpr","CastExpr","CharLiteralExpr","ClassExpr","ConditionalExpr",
                    "DoubleLiteralExpr","EnclosedExpr","FieldAccessExpr","InstanceOfExpr", "IntegerLiteralExpr",
                    "LambdaExpr","LiteralStringValueExpr","LongLiteralExpr", "MarkerAnnotationExpr","MemberValuePair",
                    "MethodCallExpr","MethodReferenceExpr","Name","NameExpr", "NormalAnnotationExpr","NullLiteralExpr",
                    "ObjectCreationExpr","PatternExpr","SimpleName", "SingleMemberAnnotationExpr","StringLiteralExpr",
                    "SuperExpr","SwitchExpr", "ThisExpr","TypeExpr","UnaryExpr","VariableDeclarationExpr","ArrayType",
                    "ClassOrInterfaceType","IntersectionType","PrimitiveType", "TypeParameter","UnionType",
                    "UnknownType","VarType","VoidType", "WildcardType","AssertStmt","BlockStmt","BreakStmt",
                    "CatchClause","ContinueStmt","DoStmt", "EmptyStmt","ExplicitConstructorInvocationStmt",
                    "ExpressionStmt","ForEachStmt","ForStmt","IfStmt", "LabeledStmt","LocalClassDeclarationStmt",
                    "LocalRecordDeclarationStmt","ReturnStmt", "SwitchEntry","SwitchStmt","ThrowStmt","TryStmt",
                    "WhileStmt", "YieldStmt","AnnotationDeclaration","AnnotationMemberDeclaration","BodyDeclaration",
                    "ClassOrInterfaceDeclaration", "ConstructorDeclaration","EnumConstantDeclaration",
                    "EnumDeclaration","FieldDeclaration","InitializerDeclaration","MethodDeclaration","Parameter",
                    "VariableDeclarator","JavadocComment", "ArrayCreationLevel","ArrayInitializerExpr")
    );

    @SuppressWarnings("unchecked")
    public static void main(String[] args) {
        JSONParser jsonParser = new JSONParser();
        try(FileReader reader = new FileReader("jsonArchives/JavaParserExample.json")) {
            Object obj =  jsonParser.parse(reader);
            JSONArray ast =  (JSONArray) obj;
            for(Object i : ast) {
                JSONObject tet =  (JSONObject) i;
                parseAST(tet);
            }

            Set<String> notAvailableInJSON = expectedASTSET.stream().filter(type -> !astTypeSet.contains(type))
                                             .collect(Collectors.toSet());
            System.out.println(notAvailableInJSON.size() +" missing Types From the Given AST\n");
            notAvailableInJSON.forEach(System.out::println);

        } catch (Exception e) {
            System.out.println("Exception Occurred - " + e);
        }
    }

    private static void parseAST(JSONObject ast) {
        String type = (String) ast.get("type");
        type = type.split("\\.")[type.split("\\.").length - 1];
        astTypeSet.add(type);
        JSONArray childArray = (JSONArray) ast.get("child");
        if (childArray == null || childArray.size() == 0) return;
        for(Object i : childArray) {
            JSONObject tet =  (JSONObject) i;
            parseAST(tet);
        }
    }
}
