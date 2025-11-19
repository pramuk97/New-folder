// DSA Solutions - Strings
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.Strings
{
    // 1. Longest Substring Without Repeating Characters
    public class LongestSubstringWithoutRepeatingCharacters
    {
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
        // Explanation: Sliding window with a set to track unique characters.
    }

    // 2. Valid Anagram
    public class ValidAnagram
    {
        public bool IsAnagram(string s, string t)
        {
            if (s.Length != t.Length) return false;
            var count = new int[26];
            foreach (var c in s) count[c - 'a']++;
            foreach (var c in t) count[c - 'a']--;
            foreach (var n in count) if (n != 0) return false;
            return true;
        }
        // Explanation: Count character frequencies and compare.
    }

    // 3. Group Anagrams
    public class GroupAnagrams
    {
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
        // Explanation: Sort each string and group by sorted value.
    }

    // 4. Longest Palindromic Substring
    public class LongestPalindromicSubstring
    {
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
        // Explanation: Expand around center for each character.
    }

    // 5. String to Integer (atoi)
    public class StringToIntegerAtoi
    {
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
        // Explanation: Parse string, handle whitespace, sign, and overflow.
    }

    // 6. Reverse Words in a String
    public class ReverseWordsInAString
    {
        public string ReverseWords(string s)
        {
            var words = s.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            Array.Reverse(words);
            return string.Join(" ", words);
        }
        // Explanation: Split, reverse, and join words.
    }

    // 7. Valid Parentheses
    public class ValidParentheses
    {
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
        // Explanation: Use stack to match opening and closing brackets.
    }

    // 8. Implement strStr()
    public class ImplementStrStr
    {
        public int StrStr(string haystack, string needle)
        {
            if (needle == "") return 0;
            for (int i = 0; i <= haystack.Length - needle.Length; i++)
            {
                if (haystack.Substring(i, needle.Length) == needle) return i;
            }
            return -1;
        }
        // Explanation: Brute force substring search.
    }

    // 9. Decode Ways
    public class DecodeWays
    {
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
        // Explanation: DP, count ways to decode using previous results.
    }

    // 10. Count and Say
    public class CountAndSay
    {
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
        // Explanation: Build next sequence by counting consecutive digits.
    }
}
