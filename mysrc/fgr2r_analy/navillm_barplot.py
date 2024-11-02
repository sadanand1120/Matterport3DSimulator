import matplotlib.pyplot as plt

# Success percentages
percentages = [94.42, 20.05, 39.13]
labels = ['R2R Train set', 'Derived Sub-instructions', 'Derived Mix and Match']

plt.figure(figsize=(8, 6))
plt.bar(labels, percentages, color='gray')
# plt.xlabel('Experiments', fontsize=14)
plt.ylabel('Success Percentage (%)', fontsize=14)
plt.title('Success Rates of NaviLLM', fontsize=16)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
