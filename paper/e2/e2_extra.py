"""Extra E2 macros from curves.csv (k where MHMG has the highest mean 3-NN accuracy; first k where L1 overtakes)."""
import pandas as pd
c = pd.read_csv("curves.csv", header=[0, 1], index_col=0)["knn_acc"]
lead = [int(k) for k in c.index if c.loc[k].idxmax() == "MHMG"]
over = [int(k) for k in c.index if k > max(lead) and c.loc[k, "L1"] > c.loc[k, "MHMG"]]
with open("e2_numbers.tex", "a") as f:
    f.write(f"\\newcommand{{\\leadLo}}{{{min(lead)}}}\n\\newcommand{{\\leadHi}}{{{max(lead)}}}\n"
            f"\\newcommand{{\\lOneFrom}}{{{min(over)}}}\n")
print(lead, min(over), all(int(k) in over for k in c.index if k >= min(over)))
