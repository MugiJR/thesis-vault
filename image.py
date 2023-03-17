import glob, json, svgwrite, os
from distinctipy import distinctipy

def getItemWidth(type):
    widths = {'com.github.javaparser.ast.expr.Name': 5, 
              'com.github.javaparser.ast.body.ClassOrInterfaceDeclaration': 5
              }
    return widths[type] if type in widths else 5
rgbs = {
        'com.github.javaparser.ast.expr.AnnotationExpr': (10, 10, 10),
        'com.github.javaparser.ast.expr.ArrayAccessExpr': (20, 20, 20),
        'com.github.javaparser.ast.expr.ArrayCreationExpr': (30, 30, 30),
        'com.github.javaparser.ast.expr.ArrayInitializerExpr': (40, 40, 40),
        'com.github.javaparser.ast.expr.AssignExpr': (50, 50, 50),
        'com.github.javaparser.ast.expr.BinaryExpr': (60, 60, 60),
        'com.github.javaparser.ast.expr.BooleanLiteralExpr': (70, 70, 70),
        'com.github.javaparser.ast.expr.CastExpr': (80, 80, 80),
        'com.github.javaparser.ast.expr.CharLiteralExpr': (90, 90, 90),
        'com.github.javaparser.ast.expr.ClassExpr': (100, 100, 100),
        'com.github.javaparser.ast.expr.ConditionalExpr': (110, 110, 110),
        'com.github.javaparser.ast.expr.DoubleLiteralExpr': (120, 120, 120),
        'com.github.javaparser.ast.expr.EnclosedExpr': (130, 130, 130),
        'com.github.javaparser.ast.expr.Expression': (140, 140, 140),
        'com.github.javaparser.ast.expr.FieldAccessExpr': (150, 150, 150),
        'com.github.javaparser.ast.expr.InstanceOfExpr': (160, 160, 160),
        'com.github.javaparser.ast.expr.IntegerLiteralExpr': (170, 170, 170),
        'com.github.javaparser.ast.expr.LambdaExpr': (180, 180, 180),
        'com.github.javaparser.ast.expr.LiteralExpr': (190, 190, 190),
        'com.github.javaparser.ast.expr.LiteralStringValueExpr': (200, 200, 200),
        'com.github.javaparser.ast.expr.LongLiteralExpr': (210, 210, 210),
        'com.github.javaparser.ast.expr.MarkerAnnotationExpr': (220, 220, 220),
        'com.github.javaparser.ast.expr.MemberValuePair': (230, 230, 240),
        'com.github.javaparser.ast.expr.MethodCallExpr': (250, 250, 250),
        'com.github.javaparser.ast.expr.MethodReferenceExpr': (0, 0, 10),
        'com.github.javaparser.ast.expr.Name': (0, 0, 20),
        'com.github.javaparser.ast.expr.NameExpr': (0, 0, 30),
        'com.github.javaparser.ast.expr.NormalAnnotationExpr': (0, 0, 40),
        'com.github.javaparser.ast.expr.NullLiteralExpr': (0, 0, 50),
        'com.github.javaparser.ast.expr.ObjectCreationExpr': (0, 0, 60),
        'com.github.javaparser.ast.expr.PatternExpr': (0, 0, 70),
        'com.github.javaparser.ast.expr.SimpleName': (0, 0, 80),
        'com.github.javaparser.ast.expr.SingleMemberAnnotationExpr': (0, 0, 90),
        'com.github.javaparser.ast.expr.StringLiteralExpr': (0, 0, 100),
        'com.github.javaparser.ast.expr.SuperExpr': (0, 0, 110),
        'com.github.javaparser.ast.expr.SwitchExpr': (0, 0, 120),
        'com.github.javaparser.ast.expr.TextBlockLiteralExpr': (0, 0, 130),
        'com.github.javaparser.ast.expr.ThisExpr': (0, 0, 140),
        'com.github.javaparser.ast.expr.TypeExpr': (0, 0, 150),
        'com.github.javaparser.ast.expr.UnaryExpr': (0, 0, 160),
        'com.github.javaparser.ast.expr.VariableDeclarationExpr': (0, 0, 170),
        'com.github.javaparser.ast.type.ArrayType': (0, 0, 180),
        'com.github.javaparser.ast.type.ArrayType.ArrayBracketPair': (0, 0, 190),
        'com.github.javaparser.ast.type.ClassOrInterfaceType': (0, 0, 200),
        'com.github.javaparser.ast.type.IntersectionType': (0, 0, 210),
        'com.github.javaparser.ast.type.PrimitiveType': (0, 0, 220),
        'com.github.javaparser.ast.type.ReferenceType': (0, 0, 230),
        'com.github.javaparser.ast.type.Type': (0, 0, 240),
        'com.github.javaparser.ast.type.TypeParameter': (0, 0, 250),
        'com.github.javaparser.ast.type.UnionType': (10, 10, 10),
        'com.github.javaparser.ast.type.UnknownType': (10, 10, 10),
        'com.github.javaparser.ast.type.VarType': (10, 10, 10),
        'com.github.javaparser.ast.type.VoidType': (10, 10, 10),
        'com.github.javaparser.ast.type.WildcardType': (10, 10, 10),
        'com.github.javaparser.ast.stmt.AssertStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.BlockStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.BreakStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.CatchClause': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ContinueStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.DoStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.EmptyStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ExplicitConstructorInvocationStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ExpressionStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ForEachStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ForStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.IfStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.LabeledStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.LocalClassDeclarationStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.LocalRecordDeclarationStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ReturnStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.Statement': (10, 10, 10),
        'com.github.javaparser.ast.stmt.SwitchEntry': (10, 10, 10),
        'com.github.javaparser.ast.stmt.SwitchStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.SynchronizedStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.ThrowStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.TryStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.UnparsableStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.WhileStmt': (10, 10, 10),
        'com.github.javaparser.ast.stmt.YieldStmt': (10, 10, 10),
        'com.github.javaparser.ast.body.AnnotationDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.AnnotationMemberDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.BodyDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.CallableDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.CallableDeclaration.Signature': (10, 10, 10),
        'com.github.javaparser.ast.body.ClassOrInterfaceDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.CompactConstructorDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.ConstructorDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.EnumConstantDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.EnumDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.FieldDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.InitializerDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.MethodDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.Parameter': (10, 10, 10),
        'com.github.javaparser.ast.body.ReceiverParameter': (10, 10, 10),
        'com.github.javaparser.ast.body.RecordDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.TypeDeclaration': (10, 10, 10),
        'com.github.javaparser.ast.body.VariableDeclarator': (10, 10, 10)}
colors = distinctipy.get_colors(len(rgbs))
def getItemColor(type):
    for i, key in enumerate(rgbs): rgbs[key] = tuple([int(colors[i][j]*256) for j in range(3)])
    return rgbs[type] if type in rgbs else (127, 127, 127)
def computeASTSizes(ast):
    astDepth, width = 0, 0
    for sibling in ast:
        if "child" in sibling:
            innerASTDepth, innerWidth = computeASTSizes(sibling["child"])
            astDepth = max(astDepth, innerASTDepth)
        else: innerWidth = 0
        #useWidth = max(innerWidth, getItemWidth(sibling["type"]) + (len(sibling["data"]) if "data" in sibling else 0))
        useWidth = max(innerWidth, getItemWidth(sibling["type"]))
        sibling['width'] = useWidth
        width += useWidth
    return astDepth+1, width

def drawAST(filename, ast, sizeInfo):
    levelHeight = 15
    # create a new SVG drawing
    dwg = svgwrite.Drawing(filename + '.svg', size=(sizeInfo[1], sizeInfo[0]*levelHeight), profile='tiny')
    def innerDrawAST(ast, baseWidth, baseHeight):
        curWidth = 0
        for sibling in ast:
            useWidth = sibling['width']
            dwg.add(dwg.rect((baseWidth+curWidth, baseHeight), (useWidth, levelHeight),
                             stroke=svgwrite.rgb(*getItemColor(sibling["type"])),
                             fill=svgwrite.rgb(*getItemColor(sibling["type"])),
                             fill_opacity=1))
            if "child" in sibling:
                innerWidth = innerDrawAST(sibling["child"], baseWidth+curWidth, baseHeight+levelHeight)
            else: innerWidth = 0
            #useWidth = max(innerWidth, getItemWidth(sibling["type"]))
            #if (sum(getItemColor(sibling["type"])) >= 256*3//2): #dark color vs light color
            #if "data" in sibling:
            #    dwg.add(dwg.text(sibling['data'], insert=(baseWidth+curWidth, baseHeight),
            #           stroke=svgwrite.rgb(255, 255, 255)))
            curWidth += useWidth        
        return curWidth
    innerDrawAST(ast, 0, 0)
    # save the SVG
    dwg.save()

if not os.path.exists("app/svgArchives"): os.mkdir("app/svgArchives")
for jsonFile in glob.glob("app/jsonArchives/*.json", recursive=True):
    with open(jsonFile, 'rb') as file:
        ast = json.load(file)
        sizeInfo = computeASTSizes(ast)
        drawAST(os.path.join("app\\svgArchives", os.path.splitext(os.path.basename(jsonFile))[0]), ast, sizeInfo)
