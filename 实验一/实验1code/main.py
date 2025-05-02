import torch
from DataProcessing import Construct_Dict
from Viterbi import HMMDecoder

def load_model():
    hanzi_list, _, hanzi_index, pinyin_index, _, pinyin2hanzi = Construct_Dict()
    decoder = HMMDecoder(
        params_path=r'F:\自然语言处理\实验一\实验1code\train_params.pt',
        hanzi_list=hanzi_list,
        pinyin2hanzi=pinyin2hanzi,
        hanzi_index=hanzi_index,
        pinyin_index=pinyin_index
    )
    return decoder

def evaluate_test_set(decoder):
    total_right = 0
    total_wrong = 0
    total_accurate = 0
    cnt = 0

    with open(r'F:\自然语言处理\实验一\实验1材料\测试集.txt', 'r', encoding='gbk') as f:
        lines = [line.strip() for line in f if line.strip()]  # 过滤空行

    # 每两行为一组：拼音行 + 正确汉字行
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines):
            break  # 确保成对

        # 读取拼音行
        pinyin_line = lines[i].casefold().split(' ')
        pinyin_seq = [p for p in pinyin_line if p]

        # 读取正确汉字行
        truth_line = lines[i+1]

        # 执行预测
        prediction = decoder.decode(pinyin_seq)
        print(f"输入拼音: {' '.join(pinyin_seq)}")
        print(f"预测结果: {prediction}")
        print(f"正确标签: {truth_line}")

        # 计算单句准确率
        right = sum(p == t for p, t in zip(prediction, truth_line))
        wrong = len(truth_line) - right
        total_right += right
        total_wrong += wrong

        # 整句是否完全正确
        if prediction == truth_line:
            total_accurate += 1

        cnt += 1
        print(f"单句汉字准确率: {right/(right+wrong):.4f}\n")

    # 输出最终统计
    print("="*50)
    print(f"测试集汉字预测准确率: {total_right/(total_right + total_wrong):.4f}")
    print(f"测试集整句预测准确率: {total_accurate/cnt:.4f}")

if __name__ == '__main__':
    decoder = load_model()
    evaluate_test_set(decoder)