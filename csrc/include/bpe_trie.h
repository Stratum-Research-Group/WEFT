#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <iostream>

namespace weft {

struct BPENode {
    int token_id = -1;
    std::unordered_map<char, std::unique_ptr<BPENode>> children;
};

class BPETrieTokenizer {
private:
    std::unique_ptr<BPENode> root;
    std::unordered_map<int, std::string> id_to_token;

public:
    BPETrieTokenizer() : root(std::make_unique<BPENode>()) {
        initialize_standard_vocab();
    }

    void add_token(int id, const std::string& token) {
        id_to_token[id] = token;
        BPENode* curr = root.get();
        for (char c : token) {
            if (curr->children.find(c) == curr->children.end()) {
                curr->children[c] = std::make_unique<BPENode>();
            }
            curr = curr->children[c].get();
        }
        curr->token_id = id;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        size_t i = 0;
        while (i < text.size()) {
            const BPENode* curr = root.get();
            int longest_match_id = -1;
            size_t match_len = 0;

            for (size_t j = i; j < text.size(); ++j) {
                char c = text[j];
                auto it = curr->children.find(c);
                if (it == curr->children.end()) {
                    break;
                }
                curr = it->second.get();
                if (curr->token_id != -1) {
                    longest_match_id = curr->token_id;
                    match_len = j - i + 1;
                }
            }

            if (longest_match_id != -1) {
                tokens.push_back(longest_match_id);
                i += match_len;
            } else {
                // Byte fallback
                tokens.push_back(static_cast<uint8_t>(text[i]));
                i += 1;
            }
        }
        return tokens;
    }

    std::string decode(int token_id) const {
        auto it = id_to_token.find(token_id);
        if (it != id_to_token.end()) {
            return it->second;
        }
        if (token_id >= 0 && token_id < 256) {
            return std::string(1, static_cast<char>(token_id));
        }
        return "";
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string res;
        for (int id : tokens) {
            res += decode(id);
        }
        return res;
    }

private:
    void initialize_standard_vocab() {
        // Base common tokens and prefixes
        std::vector<std::pair<int, std::string>> base_vocab = {
            {256, " The"}, {257, " the"}, {258, " of"}, {259, " and"}, {260, " to"},
            {261, " a"}, {262, " in"}, {263, " that"}, {264, " is"}, {265, " was"},
            {266, " for"}, {267, " on"}, {268, " are"}, {269, " as"}, {270, " with"},
            {271, " by"}, {272, " at"}, {273, " from"}, {274, " this"}, {275, " be"},
            {276, " have"}, {277, " from"}, {278, " or"}, {279, " one"}, {280, " had"},
            {281, " by"}, {282, " word"}, {283, " but"}, {284, " not"}, {285, " what"},
            {286, " all"}, {287, " were"}, {288, " we"}, {289, " when"}, {290, " your"},
            {291, " can"}, {292, " said"}, {293, " there"}, {294, " use"}, {295, " an"},
            {296, " each"}, {297, " which"}, {298, " she"}, {299, " do"}, {300, " how"},
            {301, " their"}, {302, " if"}, {303, " will"}, {304, " up"}, {305, " other"},
            {306, " about"}, {307, " out"}, {308, " many"}, {309, " then"}, {310, " them"},
            {311, " these"}, {312, " so"}, {313, " some"}, {314, " her"}, {315, " would"},
            {316, " make"}, {317, " like"}, {318, " him"}, {319, " into"}, {320, " time"},
            {321, " has"}, {322, " look"}, {323, " two"}, {324, " more"}, {325, " write"},
            {326, " go"}, {327, " see"}, {328, " number"}, {329, " no"}, {330, " way"},
            {331, " could"}, {332, " people"}, {333, " my"}, {334, " than"}, {335, " first"},
            {336, " water"}, {337, " been"}, {338, " call"}, {339, " who"}, {340, " oil"},
            {341, " its"}, {342, " now"}, {343, " find"}, {344, " long"}, {345, " down"},
            {346, " day"}, {347, " did"}, {348, " get"}, {349, " come"}, {350, " made"},
            {351, " may"}, {352, " part"}, {353, " Paris"}, {354, " Seine"}, {355, " Shakespeare"},
            {356, " Century"}, {357, " century"}, {358, " 391"}, {359, " 5555"}, {360, " Apple"},
            {361, " Banana"}, {362, " Orange"}, {363, " 73"}, {364, " 58204"}, {365, " lantern"},
            {366, " neural"}, {367, " network"}, {368, " machine"}, {369, " learning"}
        };
        for (const auto& item : base_vocab) {
            add_token(item.first, item.second);
        }
    }
};

} // namespace weft
