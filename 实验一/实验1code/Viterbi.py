import torch

class HMMDecoder:
    def __init__(self, params_path, hanzi_list, pinyin2hanzi, hanzi_index, pinyin_index):
        params = torch.load(params_path)
        self.A = params['A']  # 转移矩阵 [N, N]
        self.Pi = params['Pi']  # 初始概率 [N]
        self.hanzi_list = hanzi_list
        self.pinyin2hanzi = pinyin2hanzi
        self.hanzi_index = hanzi_index
        self.pinyin_index = pinyin_index
    
    def decode(self, pinyin_seq):
        '''维特比解码'''
        T = len(pinyin_seq)
        N = len(self.hanzi_list)
        delta = torch.full((T, N), -float('inf'))
        paths = torch.zeros((T, N), dtype=torch.long)
        
        # 初始化
        first_pinyin = pinyin_seq[0]
        possible_chars = self.pinyin2hanzi.get(first_pinyin, set())
        for char in possible_chars:
            idx = self.hanzi_index[char]
            delta[0, idx] = self.Pi[idx] * (1.0 / len(possible_chars))
        
        # 递推
        for t in range(1, T):
            curr_pinyin = pinyin_seq[t]
            possible_chars = self.pinyin2hanzi.get(curr_pinyin, set())
            for char in possible_chars:
                curr_idx = self.hanzi_index[char]
                trans_probs = delta[t-1] + torch.log(self.A[:, curr_idx] + 1e-12)
                max_prob, max_prev = torch.max(trans_probs, dim=0)
                delta[t, curr_idx] = max_prob + torch.log(torch.tensor(1.0 / len(possible_chars)))
                paths[t, curr_idx] = max_prev
        
        # 回溯路径
        best_path = torch.zeros(T, dtype=torch.long)
        best_path[-1] = torch.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            best_path[t] = paths[t+1, best_path[t+1]]
        
        # 转换为汉字
        return ''.join([self.hanzi_list[i] for i in best_path])