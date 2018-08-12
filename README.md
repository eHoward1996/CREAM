# C.R.E.A.M.

## Code Runs Everything Around Me

CREAM is the best programming language you never thought you needed.  
Inspired by the great teaching and style of the Wu-Tang Clan, CREAM  
is a dynamic powerhouse programming language. CREAM is designed to  
be the cross between Python and Java that you never wanted.  
  
## Keywords / Special Characters

CREAM (somehow) shares very little keywords with Java or Python... so pay attention  

| KeyWord / Character |       Meaning      |
|---------------------|--------------------|
|'return' | returns some value             |
|'?'      | beginning of an if statement   |
|'??'     | else if block                  |
|'?!'     | else block                     |
|'while'  | the only looping structure in CREAM because for loops are too mainstream  |
|'exit'   | same as a break statement in Java or Python exit just seemed like it made more sense  |
|'next'   | same as continue in Java or Python |
|'in'     | used as in Python's for loop (examples below) |
|'match'  | (not implemented, but it's a keyword) |
|'when'   | (not implemented, but it's a keyword) |
|'true'   | tried and true true  |
|'false'  | the opposite of true |
|'entity' | same as class identifier in Java or Python |
|'self'   | self reference (similar to Python or this in Java) |
|'->'     | denotes the beginning of a function |
|'<-'     | denotes the beginning of a class definition |

## Example Programs

### Factorial

```CREAM
factorial_with_recursion -> (n)   {
    (n == 1) ? {
        return 1
    }
    (n == 0) ?? {
        return 1
    }
    ?!  {
        return n * factorial_with_recursion(n - 1)
    }
}
.
.
.  
print("RECURSION")
print(factorial_with_recursion(10))
```

For more examples check out the `test` directory.
The entity.crm file probably won't work...but have no fear  
fixes are coming soon.