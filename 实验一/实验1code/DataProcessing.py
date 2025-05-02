import re
import torch
from collections import defaultdict

def Construct_Dict():
    '''构造汉字和拼音的映射字典'''
    hanzi_list, pinyin_list = [], []
    hanzi_index, pinyin_index = {}, {}
    hanzi2pinyin = defaultdict(set)
    pinyin2hanzi = defaultdict(set)
    
    with open(r'F:\自然语言处理\实验一\实验1材料\pinyin2hanzi.txt', 'r', encoding='UTF-8-sig') as f:
        for line in f:
            line = line.strip().split(' ')
            pinyin, chars = line[0], line[1]
            pinyin_list.append(pinyin)
            pinyin_index[pinyin] = len(pinyin_list) - 1
            
            for char in chars:
                pinyin2hanzi[pinyin].add(char)
                if char not in hanzi_index:
                    hanzi_list.append(char)
                    hanzi_index[char] = len(hanzi_list) - 1
                hanzi2pinyin[char].add(pinyin)
    
    return hanzi_list, pinyin_list, hanzi_index, pinyin_index, hanzi2pinyin, pinyin2hanzi

def Train_HMM():
    '''训练转移矩阵A和初始概率Pi'''
    hanzi_list, _, hanzi_index, _, _, _ = Construct_Dict()
    N = len(hanzi_list)
    A = torch.zeros((N, N), dtype=torch.float32)
    Pi = torch.zeros(N, dtype=torch.float32)
    
    with open(r'F:\自然语言处理\实验一\实验1材料\toutiao_cat_data.txt', 'r', encoding='UTF-8-sig') as f:
        for line in f:
            line = line.split('_!_')[3]  # 提取标题
            chars = re.findall('[\u4e00-\u9fa5]', line)
            if len(chars) < 1:
                continue
            
            # 统计初始概率
            first_char = chars[0]
            if first_char in hanzi_index:
                Pi[hanzi_index[first_char]] += 1
            
            # 统计转移概率
            for i in range(1, len(chars)):
                prev, curr = chars[i-1], chars[i]
                if prev in hanzi_index and curr in hanzi_index:
                    A[hanzi_index[prev], hanzi_index[curr]] += 1
    
    # 归一化
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    Pi = Pi / Pi.sum()
    
    # 保存为PyTorch张量
    torch.save({'A': A, 'Pi': Pi}, r'F:\自然语言处理\实验一\实验1code\train_params.pt')
    return A, Pi

if __name__ == '__main__':
    Train_HMM()
    print("训练完成")