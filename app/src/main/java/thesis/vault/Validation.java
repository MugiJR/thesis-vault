package thesis.vault;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;


public class Validation {

    private Validation(){}

    /**
     * Find given string is a valid json or not.
     * The str argument must specify a json string
     * <p>
     * This method returns true if the given string is a valid json
     * else return false
     *
     * @param  str  a json string
     * @return      true/false
     */

    public static boolean isStringValidJson(String str) {
        try {
            final ObjectMapper mapper = new ObjectMapper();
            mapper.readTree(str);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

}
