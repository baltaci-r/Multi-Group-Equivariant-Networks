import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('sh/all_results.csv')
df.drop(['finetuned_split', 'pretrain_split', 'valid_acc', 'finetuned_layer_type'], axis=1, inplace=True)
df = df[df['test_acc'].notna()]
with open('sh/final_results.csv', 'w') as f:
    for (layer, split, type), group in df.groupby(['layer_type', 'split', 'type']):
        assert len(group) == 3
        f.write(
          ','.join([layer, str(split), type, f'{group.test_acc.mean().round(4)}±{group.test_acc.std().round(4)}']
        ) + '\n')

for layer, group in df.groupby(["layer_type"]):
    group.sort_values("type", ascending=False, inplace=True)
    sns.barplot(group, x="split", y="test_acc", order=['jump',  'turn_up_jump_turn_left', 'turn_up', 'turn_left', ], hue="type", errorbar="sd", palette="tab10", saturation=1) #, order=["Pretrained", "Equitune", "Multi-Equitune"])
    plt.xlabel("Split", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    legend = plt.legend(loc='upper center', bbox_to_anchor=[0.5,1.1], ncol=3, frameon=False)
    plt.gca().set_ylim(bottom=0)
    plt.title(layer, pad=30, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"sh/SCAN_{layer}.png", dpi=100)
    plt.show()