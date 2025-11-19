# Strings DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common string-based DSA interview questions.

---

## 1. Longest Substring Without Repeating Characters
**Problem:** Find the length of the longest substring without repeating characters.

```csharp
public int LengthOfLongestSubstring(string s)
{
    var set = new HashSet<char>();
    int left = 0, maxLen = 0;
    for (int right = 0; right < s.Length; right++)
    {
        while (set.Contains(s[right]))
        {
            set.Remove(s[left++]);
        }
        set.Add(s[right]);
        maxLen = Math.Max(maxLen, right - left + 1);
    }
    return maxLen;
}
```
**Explanation:** Sliding window with a set to track unique characters.
**Time Complexity:** O(n)
**Space Complexity:** O(min(n, m))

**Dry Run Example:**
Input: s = "abcabcbb"
Step-by-step:
- right=0: set={'a'}, maxLen=1
- right=1: set={'a','b'}, maxLen=2
- right=2: set={'a','b','c'}, maxLen=3
- right=3: 'a' repeats, remove 'a', set={'b','c'}, add 'a', maxLen=3
... (continue)
Output: 3

---

## 2. Valid Anagram
**Problem:** Check if two strings are anagrams.

```csharp
public bool IsAnagram(string s, string t)
{
    if (s.Length != t.Length) return false;
    var count = new int[26];
    foreach (var c in s) count[c - 'a']++;
    foreach (var c in t) count[c - 'a']--;
    foreach (var n in count) if (n != 0) return false;
    return true;
}
```
**Explanation:** Count character frequencies and compare.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: s = "anagram", t = "nagaram"
Step-by-step:
- Count for s: [a:3, n:1, g:1, r:1, m:1]
- Subtract for t: all counts zero
Output: true

---

## 3. Group Anagrams
**Problem:** Group strings that are anagrams.

```csharp
public IList<IList<string>> GroupAnagramsMethod(string[] strs)
{
    var dict = new Dictionary<string, List<string>>();
    foreach (var str in strs)
    {
        var chars = str.ToCharArray();
        Array.Sort(chars);
        string key = new string(chars);
        if (!dict.ContainsKey(key)) dict[key] = new List<string>();
        dict[key].Add(str);
    }
    return new List<IList<string>>(dict.Values);
}
```
**Explanation:** Sort each string and group by sorted value.
**Time Complexity:** O(nk log k)
**Space Complexity:** O(nk)

**Dry Run Example:**
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Step-by-step:
- "eat" sorted: "aet" → group ["eat"]
- "tea" sorted: "aet" → group ["eat","tea"]
- ...
Output: [["eat","tea","ate"],["tan","nat"],["bat"]]

---

## 4. Longest Palindromic Substring
**Problem:** Find the longest palindromic substring.

```csharp
public string LongestPalindrome(string s)
{
    if (string.IsNullOrEmpty(s)) return "";
    int start = 0, end = 0;
    for (int i = 0; i < s.Length; i++)
    {
        int len1 = ExpandAroundCenter(s, i, i);
        int len2 = ExpandAroundCenter(s, i, i + 1);
        int len = Math.Max(len1, len2);
        if (len > end - start)
        {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.Substring(start, end - start + 1);
}
private int ExpandAroundCenter(string s, int left, int right)
{
    while (left >= 0 && right < s.Length && s[left] == s[right])
    {
        left--; right++;
    }
    return right - left - 1;
}
```
**Explanation:** Expand around center for each character.
**Time Complexity:** O(n^2)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: s = "babad"
Step-by-step:
- Center at 0: "b"
- Center at 1: "bab"
- Center at 2: "aba"
Output: "bab" or "aba"

---

## 5. String to Integer (atoi)
**Problem:** Implement atoi to convert a string to an integer.

```csharp
public int MyAtoi(string s)
{
    int i = 0, sign = 1, result = 0;
    while (i < s.Length && s[i] == ' ') i++;
    if (i < s.Length && (s[i] == '-' || s[i] == '+'))
        sign = s[i++] == '-' ? -1 : 1;
    while (i < s.Length && char.IsDigit(s[i]))
    {
        int digit = s[i++] - '0';
        if (result > (int.MaxValue - digit) / 10)
            return sign == 1 ? int.MaxValue : int.MinValue;
        result = result * 10 + digit;
    }
    return result * sign;
}
```
**Explanation:** Parse string, handle whitespace, sign, and overflow.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: s = "   -42"
Step-by-step:
- Skip spaces, sign=-1
- Read digits: 4, 2 → result=42
- Apply sign: -42
Output: -42

---

## 6. Reverse Words in a String
**Problem:** Reverse the order of words in a string.

```csharp
public string ReverseWords(string s)
{
    var words = s.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    Array.Reverse(words);
    return string.Join(" ", words);
}
```
**Explanation:** Split, reverse, and join words.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "the sky is blue"
Step-by-step:
- Split: ["the","sky","is","blue"]
- Reverse: ["blue","is","sky","the"]
- Join: "blue is sky the"
Output: "blue is sky the"

---

## 7. Valid Parentheses
**Problem:** Check if the input string is valid parentheses.

```csharp
public bool IsValid(string s)
{
    var stack = new Stack<char>();
    var map = new Dictionary<char, char> { { ')', '(' }, { '}', '{' }, { ']', '[' } };
    foreach (var c in s)
    {
        if (map.ContainsValue(c)) stack.Push(c);
        else if (map.ContainsKey(c))
        {
            if (stack.Count == 0 || stack.Pop() != map[c]) return false;
        }
    }
    return stack.Count == 0;
}
```
**Explanation:** Use stack to match opening and closing brackets.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "()[]{}"
Step-by-step:
- Push '(', pop with ')'
- Push '[', pop with ']'
- Push '{', pop with '}'
Output: true

---

## 8. Implement strStr()
**Problem:** Find the first occurrence of a substring.

```csharp
public int StrStr(string haystack, string needle)
{
    if (needle == "") return 0;
    for (int i = 0; i <= haystack.Length - needle.Length; i++)
    {
        if (haystack.Substring(i, needle.Length) == needle) return i;
    }
    return -1;
}
```
**Explanation:** Brute force substring search.
**Time Complexity:** O((n-m)m)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: haystack = "hello", needle = "ll"
Step-by-step:
- i=0: "he" != "ll"
- i=1: "el" != "ll"
- i=2: "ll" == "ll" → return 2
Output: 2

---

## 9. Decode Ways
**Problem:** Count the number of ways to decode a string.

```csharp
public int NumDecodings(string s)
{
    if (string.IsNullOrEmpty(s)) return 0;
    int n = s.Length;
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = s[0] != '0' ? 1 : 0;
    for (int i = 2; i <= n; i++)
    {
        if (s[i - 1] != '0') dp[i] += dp[i - 1];
        int twoDigit = int.Parse(s.Substring(i - 2, 2));
        if (twoDigit >= 10 && twoDigit <= 26) dp[i] += dp[i - 2];
    }
    return dp[n];
}
```
**Explanation:** DP, count ways to decode using previous results.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "226"
Step-by-step:
- dp[0]=1, dp[1]=1
- i=2: '2' valid, dp[2]=1, '22' valid, dp[2]+=1 → dp[2]=2
- i=3: '6' valid, dp[3]=2, '26' valid, dp[3]+=1 → dp[3]=3
Output: 3

---

## 10. Count and Say
**Problem:** Generate the nth term of the count-and-say sequence.

```csharp
public string CountAndSayMethod(int n)
{
    string res = "1";
    for (int i = 2; i <= n; i++)
    {
        res = Next(res);
    }
    return res;
}
private string Next(string s)
{
    var sb = new System.Text.StringBuilder();
    int count = 1;
    for (int i = 1; i <= s.Length; i++)
    {
        if (i < s.Length && s[i] == s[i - 1]) count++;
        else
        {
            sb.Append(count).Append(s[i - 1]);
            count = 1;
        }
    }
    return sb.ToString();
}
```
**Explanation:** Build next sequence by counting consecutive digits.
**Time Complexity:** O(n * m) (m = length of sequence)
**Space Complexity:** O(m)

**Dry Run Example:**
Input: n = 4
Step-by-step:
- 1: "1"
- 2: "one 1" → "11"
- 3: "two 1s" → "21"
- 4: "one 2, one 1" → "1211"
Output: "1211"

---
