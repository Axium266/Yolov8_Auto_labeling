import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import optain

def browse_weights():
    path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
    if path:
        weight_var.set(path)

def browse_input():
    path = filedialog.askdirectory()
    if path:
        input_var.set(path)

def browse_output():
    path = filedialog.askdirectory()
    if path:
        output_var.set(path)

def run():
    try:
        weights = weight_var.get()
        input_folder = input_var.get()
        output_folder = output_var.get()
        threshold_settings = {
            "none": {
                "CONF_THRESHOLD": float(none_conf_var.get()),
                "NMS_THRESHOLD": float(none_nms_var.get()),
                "IOU_THRESHOLD": float(none_iou_var.get())
            },
            "few": {
                "CONF_THRESHOLD": float(few_conf_var.get()),
                "NMS_THRESHOLD": float(few_nms_var.get()),
                "IOU_THRESHOLD": float(few_iou_var.get())
            },
            "medium": {
                "CONF_THRESHOLD": float(med_conf_var.get()),
                "NMS_THRESHOLD": float(med_nms_var.get()),
                "IOU_THRESHOLD": float(med_iou_var.get())
            },
            "many": {
                "CONF_THRESHOLD": float(many_conf_var.get()),
                "NMS_THRESHOLD": float(many_nms_var.get()),
                "IOU_THRESHOLD": float(many_iou_var.get())
            }
        }
        few_max = int(few_max_var.get())
        med_max = int(med_max_var.get())
    except Exception as e:
        messagebox.showerror("錯誤", f"輸入錯誤: {e}")
        return

    if not weights or not input_folder or not output_folder:
        messagebox.showwarning("提醒", "請完整填寫模型權重、輸入資料夾與輸出資料夾")
        return

    try:
        optain.run(weights, input_folder, output_folder, threshold_settings, few_max, med_max)
        messagebox.showinfo("完成", "所有圖片標註完成！")
    except Exception as ex:
        messagebox.showerror("錯誤", f"標註失敗: {ex}")

root = tk.Tk()
root.title("YOLOv8自動標註")
root.geometry("720x440")

weight_var = tk.StringVar()
input_var = tk.StringVar()
output_var = tk.StringVar()

none_conf_var = tk.StringVar(value="0.3")
none_nms_var = tk.StringVar(value="0.4")
none_iou_var = tk.StringVar(value="0.4")

few_conf_var = tk.StringVar(value="0.5")
few_nms_var = tk.StringVar(value="0.3")
few_iou_var = tk.StringVar(value="0.3")

med_conf_var = tk.StringVar(value="0.6")
med_nms_var = tk.StringVar(value="0.25")
med_iou_var = tk.StringVar(value="0.25")

many_conf_var = tk.StringVar(value="0.7")
many_nms_var = tk.StringVar(value="0.2")
many_iou_var = tk.StringVar(value="0.2")

few_max_var = tk.StringVar(value="5")
med_max_var = tk.StringVar(value="10")

# 置放輸入欄與檔案選擇
tk.Label(root, text="模型權重(.pt)：").grid(row=0, column=0, sticky="e", pady=10, padx=5)
tk.Entry(root, textvariable=weight_var, width=60).grid(row=0, column=1, padx=5)
tk.Button(root, text="選擇模型", command=browse_weights).grid(row=0, column=2, padx=5)

tk.Label(root, text="輸入圖片資料夾：").grid(row=1, column=0, sticky="e", pady=10, padx=5)
tk.Entry(root, textvariable=input_var, width=60).grid(row=1, column=1, padx=5)
tk.Button(root, text="選擇資料夾", command=browse_input).grid(row=1, column=2, padx=5)

tk.Label(root, text="輸出資料夾：").grid(row=2, column=0, sticky="e", pady=10, padx=5)
tk.Entry(root, textvariable=output_var, width=60).grid(row=2, column=1, padx=5)
tk.Button(root, text="選擇資料夾", command=browse_output).grid(row=2, column=2, padx=5)

# 門檻分段設定
tk.Label(root, text="few最大標註數量 (含)：").grid(row=3, column=0, sticky="e", pady=10, padx=5)
tk.Entry(root, textvariable=few_max_var, width=8).grid(row=3, column=1, sticky="w")

tk.Label(root, text="medium最大標註數量 (含)：").grid(row=4, column=0, sticky="e", pady=10, padx=5)
tk.Entry(root, textvariable=med_max_var, width=8).grid(row=4, column=1, sticky="w")

# 隔線
separator = ttk.Separator(root, orient='horizontal')
separator.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(15,10))

# 模擬表格放入frame讓各欄寬自適應
table_frame = tk.Frame(root)
table_frame.grid(row=6, column=0, columnspan=3, sticky="w", padx=10)

tk.Label(table_frame, text=" ").grid(row=0, column=0)
tk.Label(table_frame, text="CONF").grid(row=0, column=1)
tk.Label(table_frame, text="NMS").grid(row=0, column=2)
tk.Label(table_frame, text="IOU").grid(row=0, column=3)

categories = ["none", "few", "medium", "many"]
conf_vars = [none_conf_var, few_conf_var, med_conf_var, many_conf_var]
nms_vars = [none_nms_var, few_nms_var, med_nms_var, many_nms_var]
iou_vars = [none_iou_var, few_iou_var, med_iou_var, many_iou_var]

for i, cat in enumerate(categories):
    tk.Label(table_frame, text=cat).grid(row=i+1, column=0, sticky="w")
    tk.Entry(table_frame, textvariable=conf_vars[i], width=8, justify="center").grid(row=i+1, column=1)
    tk.Entry(table_frame, textvariable=nms_vars[i], width=8, justify="center").grid(row=i+1, column=2)
    tk.Entry(table_frame, textvariable=iou_vars[i], width=8, justify="center").grid(row=i+1, column=3)

tk.Button(root, text="開始標註", width=25, command=run).grid(row=7 + len(categories), column=1, pady=30)

root.mainloop()