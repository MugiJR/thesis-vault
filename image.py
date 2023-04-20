import glob
import json
import svgwrite
import os
from distinctipy import distinctipy

javaparserkeys = ['com.github.javaparser.ast.expr.AnnotationExpr', 'com.github.javaparser.ast.expr.ArrayAccessExpr', 'com.github.javaparser.ast.expr.ArrayCreationExpr', 'com.github.javaparser.ast.expr.ArrayInitializerExpr', 'com.github.javaparser.ast.expr.AssignExpr', 'com.github.javaparser.ast.expr.BinaryExpr', 'com.github.javaparser.ast.expr.BooleanLiteralExpr', 'com.github.javaparser.ast.expr.CastExpr', 'com.github.javaparser.ast.expr.CharLiteralExpr', 'com.github.javaparser.ast.expr.ClassExpr', 'com.github.javaparser.ast.expr.ConditionalExpr', 'com.github.javaparser.ast.expr.DoubleLiteralExpr', 'com.github.javaparser.ast.expr.EnclosedExpr', 'com.github.javaparser.ast.expr.Expression', 'com.github.javaparser.ast.expr.FieldAccessExpr', 'com.github.javaparser.ast.expr.InstanceOfExpr', 'com.github.javaparser.ast.expr.IntegerLiteralExpr', 'com.github.javaparser.ast.expr.LambdaExpr', 'com.github.javaparser.ast.expr.LiteralExpr', 'com.github.javaparser.ast.expr.LiteralStringValueExpr', 'com.github.javaparser.ast.expr.LongLiteralExpr', 'com.github.javaparser.ast.expr.MarkerAnnotationExpr', 'com.github.javaparser.ast.expr.MemberValuePair', 'com.github.javaparser.ast.expr.MethodCallExpr', 'com.github.javaparser.ast.expr.MethodReferenceExpr', 'com.github.javaparser.ast.expr.Name',
                  'com.github.javaparser.ast.expr.NameExpr', 'com.github.javaparser.ast.expr.NormalAnnotationExpr', 'com.github.javaparser.ast.expr.NullLiteralExpr', 'com.github.javaparser.ast.expr.ObjectCreationExpr', 'com.github.javaparser.ast.expr.PatternExpr', 'com.github.javaparser.ast.expr.SimpleName', 'com.github.javaparser.ast.expr.SingleMemberAnnotationExpr', 'com.github.javaparser.ast.expr.StringLiteralExpr', 'com.github.javaparser.ast.expr.SuperExpr', 'com.github.javaparser.ast.expr.SwitchExpr', 'com.github.javaparser.ast.expr.TextBlockLiteralExpr', 'com.github.javaparser.ast.expr.ThisExpr', 'com.github.javaparser.ast.expr.TypeExpr', 'com.github.javaparser.ast.expr.UnaryExpr', 'com.github.javaparser.ast.expr.VariableDeclarationExpr', 'com.github.javaparser.ast.type.ArrayType', 'com.github.javaparser.ast.type.ArrayType.ArrayBracketPair', 'com.github.javaparser.ast.type.ClassOrInterfaceType', 'com.github.javaparser.ast.type.IntersectionType', 'com.github.javaparser.ast.type.PrimitiveType', 'com.github.javaparser.ast.type.ReferenceType', 'com.github.javaparser.ast.type.Type', 'com.github.javaparser.ast.type.TypeParameter', 'com.github.javaparser.ast.type.UnionType', 'com.github.javaparser.ast.type.UnknownType', 'com.github.javaparser.ast.type.VarType', 'com.github.javaparser.ast.type.VoidType', 'com.github.javaparser.ast.type.WildcardType', 'com.github.javaparser.ast.stmt.AssertStmt', 'com.github.javaparser.ast.stmt.BlockStmt', 'com.github.javaparser.ast.stmt.BreakStmt', 'com.github.javaparser.ast.stmt.CatchClause', 'com.github.javaparser.ast.stmt.ContinueStmt', 'com.github.javaparser.ast.stmt.DoStmt', 'com.github.javaparser.ast.stmt.EmptyStmt', 'com.github.javaparser.ast.stmt.ExplicitConstructorInvocationStmt', 'com.github.javaparser.ast.stmt.ExpressionStmt', 'com.github.javaparser.ast.stmt.ForEachStmt', 'com.github.javaparser.ast.stmt.ForStmt', 'com.github.javaparser.ast.stmt.IfStmt', 'com.github.javaparser.ast.stmt.LabeledStmt', 'com.github.javaparser.ast.stmt.LocalClassDeclarationStmt', 'com.github.javaparser.ast.stmt.LocalRecordDeclarationStmt', 'com.github.javaparser.ast.stmt.ReturnStmt', 'com.github.javaparser.ast.stmt.Statement', 'com.github.javaparser.ast.stmt.SwitchEntry', 'com.github.javaparser.ast.stmt.SwitchStmt', 'com.github.javaparser.ast.stmt.SynchronizedStmt', 'com.github.javaparser.ast.stmt.ThrowStmt', 'com.github.javaparser.ast.stmt.TryStmt', 'com.github.javaparser.ast.stmt.UnparsableStmt', 'com.github.javaparser.ast.stmt.WhileStmt', 'com.github.javaparser.ast.stmt.YieldStmt', 'com.github.javaparser.ast.body.AnnotationDeclaration', 'com.github.javaparser.ast.body.AnnotationMemberDeclaration', 'com.github.javaparser.ast.body.BodyDeclaration', 'com.github.javaparser.ast.body.CallableDeclaration', 'com.github.javaparser.ast.body.CallableDeclaration.Signature', 'com.github.javaparser.ast.body.ClassOrInterfaceDeclaration', 'com.github.javaparser.ast.body.CompactConstructorDeclaration', 'com.github.javaparser.ast.body.ConstructorDeclaration', 'com.github.javaparser.ast.body.EnumConstantDeclaration', 'com.github.javaparser.ast.body.EnumDeclaration', 'com.github.javaparser.ast.body.FieldDeclaration', 'com.github.javaparser.ast.body.InitializerDeclaration', 'com.github.javaparser.ast.body.MethodDeclaration', 'com.github.javaparser.ast.body.Parameter', 'com.github.javaparser.ast.body.ReceiverParameter', 'com.github.javaparser.ast.body.RecordDeclaration', 'com.github.javaparser.ast.body.TypeDeclaration', 'com.github.javaparser.ast.body.VariableDeclarator']

def getItemWidth(type):
    widths = {'com.github.javaparser.ast.expr.Name': 5,
              'com.github.javaparser.ast.body.ClassOrInterfaceDeclaration': 5
              }
    return widths[type] if type in widths else 5

rgbs = {
    'com.github.javaparser.ast.expr.AnnotationExpr': (0, 256, 0),
    'com.github.javaparser.ast.expr.ArrayAccessExpr': (256, 0, 256),
    'com.github.javaparser.ast.expr.ArrayCreationExpr': (0, 128, 256),
    'com.github.javaparser.ast.expr.ArrayInitializerExpr': (256, 128, 0),
    'com.github.javaparser.ast.expr.AssignExpr': (128, 192, 128),
    'com.github.javaparser.ast.expr.BinaryExpr': (86, 8, 171),
    'com.github.javaparser.ast.expr.BooleanLiteralExpr': (196, 4, 24),
    'com.github.javaparser.ast.expr.CastExpr': (39, 127, 2),
    'com.github.javaparser.ast.expr.CharLiteralExpr': (255, 122, 190),
    'com.github.javaparser.ast.expr.ClassExpr': (0, 256, 256),
    'com.github.javaparser.ast.expr.ConditionalExpr': (256, 256, 0),
    'com.github.javaparser.ast.expr.DoubleLiteralExpr': (0, 256, 128),
    'com.github.javaparser.ast.expr.EnclosedExpr': (0, 128, 128),
    'com.github.javaparser.ast.expr.Expression': (139, 92, 87),
    'com.github.javaparser.ast.expr.FieldAccessExpr': (136, 153, 251),
    'com.github.javaparser.ast.expr.InstanceOfExpr': (126, 228, 10),
    'com.github.javaparser.ast.expr.IntegerLiteralExpr': (249, 211, 121),
    'com.github.javaparser.ast.expr.LambdaExpr': (128, 256, 256),
    'com.github.javaparser.ast.expr.LiteralExpr': (0, 0, 256),
    'com.github.javaparser.ast.expr.LiteralStringValueExpr': (199, 24, 151),
    'com.github.javaparser.ast.expr.LongLiteralExpr': (142, 67, 226),
    'com.github.javaparser.ast.expr.MarkerAnnotationExpr': (12, 59, 74),
    'com.github.javaparser.ast.expr.MemberValuePair': (31, 192, 195),
    'com.github.javaparser.ast.expr.MethodCallExpr': (45, 193, 61),
    'com.github.javaparser.ast.expr.MethodReferenceExpr': (158, 142, 11),
    'com.github.javaparser.ast.expr.Name': (99, 6, 59),
    'com.github.javaparser.ast.expr.NameExpr': (235, 82, 83),
    'com.github.javaparser.ast.expr.NormalAnnotationExpr': (87, 116, 174),
    'com.github.javaparser.ast.expr.NullLiteralExpr': (0, 0, 128),
    'com.github.javaparser.ast.expr.ObjectCreationExpr': (185, 253, 164),
    'com.github.javaparser.ast.expr.PatternExpr': (50, 63, 245),
    'com.github.javaparser.ast.expr.SimpleName': (102, 68, 4),
    'com.github.javaparser.ast.expr.SingleMemberAnnotationExpr': (227, 177, 252),
    'com.github.javaparser.ast.expr.StringLiteralExpr': (128, 0, 256),
    'com.github.javaparser.ast.expr.SuperExpr': (175, 138, 164),
    'com.github.javaparser.ast.expr.SwitchExpr': (102, 254, 108),
    'com.github.javaparser.ast.expr.TextBlockLiteralExpr': (6, 64, 174),
    'com.github.javaparser.ast.expr.ThisExpr': (217, 191, 1),
    'com.github.javaparser.ast.expr.TypeExpr': (243, 63, 240),
    'com.github.javaparser.ast.expr.UnaryExpr': (190, 78, 5),
    'com.github.javaparser.ast.expr.VariableDeclarationExpr': (225, 149, 101),
    'com.github.javaparser.ast.type.ArrayType': (204, 250, 65),
    'com.github.javaparser.ast.type.ArrayType.ArrayBracketPair': (58, 253, 198),
    'com.github.javaparser.ast.type.ClassOrInterfaceType': (1, 65, 1),
    'com.github.javaparser.ast.type.IntersectionType': (92, 143, 81),
    'com.github.javaparser.ast.type.PrimitiveType': (250, 21, 80),
    'com.github.javaparser.ast.type.ReferenceType': (116, 208, 206),
    'com.github.javaparser.ast.type.Type': (197, 109, 249),
    'com.github.javaparser.ast.type.TypeParameter': (26, 192, 128),
    'com.github.javaparser.ast.type.UnionType': (190, 202, 195),
    'com.github.javaparser.ast.type.UnknownType': (85, 56, 116),
    'com.github.javaparser.ast.type.VarType': (169, 194, 62),
    'com.github.javaparser.ast.type.VoidType': (185, 79, 167),
    'com.github.javaparser.ast.type.WildcardType': (82, 182, 1),
    'com.github.javaparser.ast.stmt.AssertStmt': (2, 181, 6),
    'com.github.javaparser.ast.stmt.BlockStmt': (81, 115, 254),
    'com.github.javaparser.ast.stmt.BreakStmt': (2, 114, 64),
    'com.github.javaparser.ast.stmt.CatchClause': (190, 1, 225),
    'com.github.javaparser.ast.stmt.ContinueStmt': (255, 51, 9),
    'com.github.javaparser.ast.stmt.DoStmt': (137, 14, 120),
    'com.github.javaparser.ast.stmt.EmptyStmt': (176, 45, 68),
    'com.github.javaparser.ast.stmt.ExplicitConstructorInvocationStmt': (44, 249, 57),
    'com.github.javaparser.ast.stmt.ExpressionStmt': (253, 249, 180),
    'com.github.javaparser.ast.stmt.ForEachStmt': (71, 180, 254),
    'com.github.javaparser.ast.stmt.ForStmt': (255, 63, 160),
    'com.github.javaparser.ast.stmt.IfStmt': (128, 0, 0),
    'com.github.javaparser.ast.stmt.LabeledStmt': (191, 246, 233),
    'com.github.javaparser.ast.stmt.LocalClassDeclarationStmt': (256, 0, 0),
    'com.github.javaparser.ast.stmt.LocalRecordDeclarationStmt': (5, 9, 193),
    'com.github.javaparser.ast.stmt.ReturnStmt': (15, 128, 190),
    'com.github.javaparser.ast.stmt.Statement': (251, 0, 176),
    'com.github.javaparser.ast.stmt.SwitchEntry': (88, 58, 190),
    'com.github.javaparser.ast.stmt.SwitchStmt': (29, 8, 62),
    'com.github.javaparser.ast.stmt.SynchronizedStmt': (241, 173, 171),
    'com.github.javaparser.ast.stmt.ThrowStmt': (208, 119, 44),
    'com.github.javaparser.ast.stmt.TryStmt': (74, 90, 58),
    'com.github.javaparser.ast.stmt.UnparsableStmt': (92, 164, 166),
    'com.github.javaparser.ast.stmt.WhileStmt': (70, 228, 148),
    'com.github.javaparser.ast.stmt.YieldStmt': (69, 7, 230),
    'com.github.javaparser.ast.body.AnnotationDeclaration': (184, 212, 128),
    'com.github.javaparser.ast.body.AnnotationMemberDeclaration': (128, 128, 128),
    'com.github.javaparser.ast.body.BodyDeclaration': (60, 28, 6),
    'com.github.javaparser.ast.body.CallableDeclaration': (161, 199, 254),
    'com.github.javaparser.ast.body.CallableDeclaration.Signature': (8, 203, 253),
    'com.github.javaparser.ast.body.ClassOrInterfaceDeclaration': (146, 17, 181),
    'com.github.javaparser.ast.body.CompactConstructorDeclaration': (111, 215, 71),
    'com.github.javaparser.ast.body.ConstructorDeclaration': (70, 102, 118),
    'com.github.javaparser.ast.body.EnumConstantDeclaration': (160, 151, 87),
    'com.github.javaparser.ast.body.EnumDeclaration': (192, 110, 111),
    'com.github.javaparser.ast.body.FieldDeclaration': (255, 203, 48),
    'com.github.javaparser.ast.body.InitializerDeclaration': (256, 128, 256),
    'com.github.javaparser.ast.body.MethodDeclaration': (100, 132, 16),
    'com.github.javaparser.ast.body.Parameter': (62, 248, 4),
    'com.github.javaparser.ast.body.ReceiverParameter': (192, 246, 0),
    'com.github.javaparser.ast.body.RecordDeclaration': (146, 115, 209),
    'com.github.javaparser.ast.body.TypeDeclaration': (2, 159, 85),
    'com.github.javaparser.ast.body.VariableDeclarator': (121, 174, 46)
}

# colors = distinctipy.get_colors(len(rgbs))
#print({key: tuple(int(x*256) for x in color) for key, color in zip(javaparserkeys, distinctipy.get_colors(len(javaparserkeys)))})
#print(distinctipy.get_colors(len(rgbs)))


def getItemColor(type):
    return rgbs.get(type) if type in rgbs else (127, 127, 127)
    # for i, key in enumerate(rgbs):
    #     rgbs[key] = tuple([int(colors[i][j]*256) for j in range(3)])
    # return rgbs[type] if type in rgbs else (127, 127, 127)

    


def computeASTSizes(ast):
    astDepth, width = 0, 0
    for sibling in ast:
        if "child" in sibling:
            innerASTDepth, innerWidth = computeASTSizes(sibling["child"])
            astDepth = max(astDepth, innerASTDepth)
        else:
            innerWidth = 0
        # useWidth = max(innerWidth, getItemWidth(sibling["type"]) + (len(sibling["data"]) if "data" in sibling else 0))
        useWidth = max(innerWidth, getItemWidth(sibling["type"]))
        sibling['width'] = useWidth
        width += useWidth
    return astDepth+1, width


def drawAST(filename, pngname, ast, sizeInfo):
    levelHeight = 15
    # create a new SVG drawing
    dwg = svgwrite.Drawing(
        filename + '.svg', size=(sizeInfo[1], sizeInfo[0]*levelHeight), profile='tiny')

    def innerDrawAST(ast, baseWidth, baseHeight):
        curWidth = 0
        for sibling in ast:
            useWidth = sibling['width']
            dwg.add(dwg.rect((baseWidth+curWidth, baseHeight), (useWidth, levelHeight),
                             stroke=svgwrite.rgb(*getItemColor(sibling["type"])),
                             fill=svgwrite.rgb(*getItemColor(sibling["type"])),
                             fill_opacity=1))
            if "child" in sibling:
                innerWidth = innerDrawAST(
                    sibling["child"], baseWidth+curWidth, baseHeight+levelHeight)
            else:
                innerWidth = 0
            # useWidth = max(innerWidth, getItemWidth(sibling["type"]))
            # if (sum(getItemColor(sibling["type"])) >= 256*3//2): #dark color vs light color
            # if "data" in sibling:
            #    dwg.add(dwg.text(sibling['data'], insert=(baseWidth+curWidth, baseHeight),
            #           stroke=svgwrite.rgb(255, 255, 255)))
            curWidth += useWidth
        return curWidth
    innerDrawAST(ast, 0, 0)
    # save the SVG
    dwg.save()
    #install Inkscape and/or GraphViz for Windows
    #from cairosvg import svg2png
    #svg2png(file_obj=open(filename + ".svg", "rb"),write_to=filename + '.png')
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    drawing = svg2rlg(filename + ".svg")
    renderPM.drawToFile(drawing, pngname + ".png", fmt="PNG")

if not os.path.exists("app/svgArchives"): os.mkdir("app/svgArchives")
if not os.path.exists("app/pngArchives"): os.mkdir("app/pngArchives")
for label in range(1, 5+1):
    if not os.path.exists("app/svgArchives" + "/" + str(label)): os.mkdir("app/svgArchives" + "/" + str(label))
    if not os.path.exists("app/pngArchives" + "/" + str(label)): os.mkdir("app/pngArchives" + "/" + str(label))
for jsonFile in glob.glob("app/jsonArchives/*.json", recursive=True):
        with open(jsonFile, 'rb') as file:
            ast = json.load(file)
            sizeInfo = computeASTSizes(ast)
        label = 1
        drawAST(os.path.join("app\\svgArchives", str(label), os.path.splitext(os.path.basename(jsonFile))[0]),
                os.path.join("app\\pngArchives", str(label), os.path.splitext(os.path.basename(jsonFile))[0]),
                ast, sizeInfo)
