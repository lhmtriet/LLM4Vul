from tree_sitter import Language, Parser


class TSParse:
    r'''Parse C/C++ code using tree-sitter AST.
    Example:
    code = """
        #include "ap_config.h"
        using namespace ContosoData;\n
        int main() {
            int a = 1; // comment
            /* comment // comment /*
            int r += 4;
            here */ int c = 2;
            int d = 4;
            int b /* comment */ = 2;
            int e = 5;
        }
    """
    print(TSParse("./treesitter_c_cpp.so").noisy(code, "cpp", True))
    '''
    # Tung: Add lang parameter list_languages
    def __init__(self, treesitter_path="./treesitter_c_cpp.so", 
                 list_languages=None,
                 treesitter_folder=None):
        """Initialize C/C++ Tree-Sitter parsers."""
        self.parsers = {}
        self.langs = {}
        self.noisy_query = {}
        
        # Tung: comments and edit this code below
        # for lang in ["c", "cpp"]:
        #     self.langs[lang] = Language(treesitter_path, lang)
        #     parser = Parser()
        #     parser.set_language(self.langs[lang])
        #     self.parsers[lang] = parser
        
        if type(list_languages) is list:
            for lang in list_languages:
                language_folder = f"{treesitter_folder}/{lang}.so"
                self.langs[lang] = Language(language_folder, lang)
                parser = Parser()
                parser.set_language(self.langs[lang])
                self.parsers[lang] = parser
        else:
            for lang in ["c", "cpp"]:
                self.langs[lang] = Language(treesitter_path, lang)
                parser = Parser()
                parser.set_language(self.langs[lang])
                self.parsers[lang] = parser
        
        self.noisy_query[
            "cpp"
        ] = """
            (comment) @comment
            (preproc_include) @include
            (using_declaration) @using
        """

        self.noisy_query["c"] = "(comment) @comment (preproc_include) @include"
        
        
        # Tung: Add noisy query for Rust
        self.noisy_query["rust"] = """
            (line_comment) @comment
            (block_comment) @comment
        """
        
        # Tung: Add noisy query for Swift
        self.noisy_query["swift"] = """
            (line_comment) @comment
            (block_comment) @comment
        """
        
        # Tung: Add noisy query for Kotlin
        self.noisy_query["kotlin"] = """
            (line_comment) @comment
        """

    def normlang(self, lang):
        """Normalize language."""
        lang = lang.lower()
        lang = "cpp" if lang == "c++" else lang
        return lang

    def query(self, code, lang, query):
        """Run query on code."""
        # Get AST
        lang = self.normlang(lang)
        tree = self.parsers[lang].parse(bytes(code, "utf8"))

        # Return results
        results = self.langs[lang].query(query).captures(tree.root_node)
        # print(results)
        return results

    def clean(self, code, lang):
        """Remove comments and includes from code.
        Args:
            code (str): Code as a string with newlines
            lang (str): Can be C or C++
        """
        # Find comments and includes
        lang = self.normlang(lang)
        results = self.query(code, lang, self.noisy_query[lang])

        # print(results)

        # Remove comments and includes based on string slicing
        code_new = code.splitlines()
        for i in results:
            sline, schar = i[0].start_point
            eline, echar = i[0].end_point
            # print('I=', i)
            if sline == eline:
                code_new[sline] = code_new[sline][:schar] + code_new[sline][echar:]
            else:
                for line_index in range(sline, min(len(code_new), eline + 1)):
                    if line_index == sline:
                        code_new[sline] = code_new[sline][:schar]
                    elif line_index == eline:
                        code_new[eline] = code_new[eline][echar:]
                    else:
                        code_new[line_index] = ""
            # print(code_new)
        # print(code_new, len(code_new))
        # print(len(code_new))
        return "\n".join(code_new)

    def noisy(self, code, lang, debug=False):
        """Return noisy lines from code."""
        # Find comments and includes
        lang = self.normlang(lang)
        results = self.query(code, lang, self.noisy_query[lang])
        ret = []
        for i in results:
            for j in range(i[0].start_point[0], i[0].end_point[0] + 1):
                # print(j)
                ret += [j]
        
        code = code.splitlines()
        ret += [c for c, i in enumerate(code) if len(i) == 0 or i.isspace()]
        ret = set(ret)
        
        if debug:
            for i in range(len(code)):
                if i in ret:
                    code[i] = "# " + code[i]
            return "\n".join(code)
        return list(ret)
        
    def noisy_lines(self, code, lang):
        clean_code = self.clean(code, lang)
        
        tmp_code = clean_code.splitlines()
        ret = []
        ret += [c for c, line in enumerate(tmp_code) if len(line) == 0 or line.isspace() or line == '']
        return list(set(ret)), clean_code


