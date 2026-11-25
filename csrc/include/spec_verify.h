// Speculative tree verification (greedy) + minimal JSON int-array extraction.
// Semantics mirror weft/speculative_tree.py::SpeculativeTreeEngine.verify_tree:
//   node 0 accepted iff draft[0]==target[0]; node i accepted iff parent[i] is the most recently
//   accepted node and draft[i]==target[i]; first failure stops verification.
// If `parents` is empty the draft is treated as a chain (parent[i]=i-1).
#pragma once
#include <string>
#include <vector>
#include <cstdlib>

namespace specv {

struct Result {
    std::vector<long> accepted;
    int diverged_index;     // index of first rejected node, or -1 if all accepted
    bool ok;                // false => malformed request
    std::string error;
};

// Find `"key": [ints]` inside [begin,end) of s. Returns false if key missing or malformed.
inline bool get_int_array(const std::string& s, const std::string& key, std::vector<long>& out,
                          size_t begin = 0, size_t end = std::string::npos) {
    if (end == std::string::npos || end > s.size()) end = s.size();
    std::string needle = "\"" + key + "\"";
    size_t k = s.find(needle, begin);
    if (k == std::string::npos || k >= end) return false;
    size_t lb = s.find('[', k);
    if (lb == std::string::npos || lb >= end) return false;
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos || rb >= end) return false;
    out.clear();
    size_t p = lb + 1;
    while (p < rb) {
        while (p < rb && (s[p] == ' ' || s[p] == ',' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
        if (p >= rb) break;
        char* endp = nullptr;
        long v = std::strtol(s.c_str() + p, &endp, 10);
        if (endp == s.c_str() + p) return false;  // not a number
        out.push_back(v);
        p = endp - s.c_str();
    }
    return true;
}

inline Result verify(const std::vector<long>& draft, const std::vector<long>& target,
                     const std::vector<long>& parents) {
    Result r{{}, -1, true, ""};
    if (draft.empty()) { r.ok = false; r.error = "draft_tokens missing or empty"; return r; }
    if (target.size() < draft.size()) { r.ok = false; r.error = "target_tokens shorter than draft_tokens"; return r; }
    if (!parents.empty() && parents.size() != draft.size()) { r.ok = false; r.error = "parents length != draft length"; return r; }
    long last_accepted_node = -1;
    for (size_t i = 0; i < draft.size(); i++) {
        long parent = parents.empty() ? (long)i - 1 : parents[i];
        bool parent_ok = (parent == last_accepted_node);
        if (parent_ok && draft[i] == target[i]) {
            r.accepted.push_back(draft[i]);
            last_accepted_node = (long)i;
        } else {
            r.diverged_index = (int)i;
            return r;
        }
    }
    return r;
}

inline std::string list_json(const std::vector<long>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) { if (i) s += ","; s += std::to_string(v[i]); }
    return s + "]";
}

inline std::string result_json(const Result& r) {
    return "\"accepted_count\":" + std::to_string(r.accepted.size()) +
           ",\"accepted_tokens\":" + list_json(r.accepted) +
           ",\"diverged_index\":" + std::to_string(r.diverged_index);
}

inline std::string body_of(const std::string& req) {
    size_t h = req.find("\r\n\r\n");
    return h == std::string::npos ? std::string() : req.substr(h + 4);
}

// Split "streams":[ {...},{...} ] into object substrings via brace matching.
inline std::vector<std::string> split_stream_objects(const std::string& body) {
    std::vector<std::string> objs;
    size_t k = body.find("\"streams\"");
    if (k == std::string::npos) return objs;
    size_t p = body.find('[', k);
    if (p == std::string::npos) return objs;
    int depth = 0; size_t start = 0;
    for (size_t i = p + 1; i < body.size(); i++) {
        if (body[i] == '{') { if (depth == 0) start = i; depth++; }
        else if (body[i] == '}') { depth--; if (depth == 0) objs.push_back(body.substr(start, i - start + 1)); }
        else if (body[i] == ']' && depth == 0) break;
    }
    return objs;
}

}  // namespace specv
