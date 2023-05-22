/**
 * This is a simple Java program that brief about JavaParser's.
 */
import java.io.Serializable;
import java.lang.annotation.*;
import java.util.*;

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface MyAnnotation { String value(); }
interface MyInterface { void myMethod(); }
enum MyEnum { ENUM_CONSTANT; }
@MyAnnotation(value = "java")
public class JavaParserExample <T extends Serializable & Cloneable> {
    @Override public String toString() { return super.toString(); }
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};
        System.out.println(numbers[0]);
        int[][] matrix = new int[3][3];
        String[] names = new String[]{"Alice", "Bob", "Charlie", null};
        int result = 10 + 5;
        boolean isTrue = (result > 10) && names instanceof String[];
        String resultMessage = "Result is greater than 10";
        Object strObj = "java";
        for(int i=0; i<5; i++) {}
        firstLbl:
            for (;result<18;result++) {
                if (isTrue && strObj instanceof String s) {
                    System.out.println(resultMessage);
                    break firstLbl;
                } else {
                    System.out.println("Result is less than or equal to 10");
                    assert result > 1;
                    continue firstLbl;
                }
            }
        while(result > 20) {
            matrix[0][0] = result;
            result++;
        }
        char letter = 'A';
        double pi = 3.14;
        boolean bool = true;
        long largeNumber = 1234567890L;
        int smallNumber = (int) pi;
        int a = switch(smallNumber) { case 5,6: yield 20; default: yield 5+5; };
        switch (smallNumber) {
            case 1: bool = false; break;
            default: break;
        }
        String message = isTrue ? "True" : "False";
        List<Integer> numbersList = Arrays.asList(1, 2, 3, 4, 5);
        numbersList.stream().mapToDouble(Double::valueOf).forEach(n -> System.out.println(n));
        MyClass obj2 = new MyClass();
        try{
            obj2.myMethod();
        }catch (ArrayIndexOutOfBoundsException | NullPointerException e) { e.printStackTrace();}
        int x = 10;
        int y = x++;
    }

    static class MyClass {
        int a;
        MyClass(){super(); this.a = 10; class MyClassHelper { }}
        static <T> void myMethod() throws ArrayIndexOutOfBoundsException, NullPointerException {
            System.out.println("Generic Method: ");
        }
        void printCollection(Collection<?> c) throws Exception {
            do {
                for(Object t : c) continue;
            } while (c.size() == 0);
            throw new Exception();
        }
        record MyRecordCollection() {}
    }
}

