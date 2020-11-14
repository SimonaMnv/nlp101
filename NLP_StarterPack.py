import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import NLP.NLP_SpamDetection
import NLP.NLP_RestaurantReviewClassification2
import NLP.NLP_SpamDetectionCNN
from PIL import Image, ImageTk
from tkinter import Label, Toplevel, LEFT, SOLID


class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


# set up tkinter basics
root = tk.Tk()
root.geometry("600x600")
ttk.Style().theme_use('clam')
root.pack_propagate(False)  # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0)  # fixed  size.

# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Results of Analysis")
frame1.place(height=380, width=600)
frame1.master.title("NLP Starter Pack")

# Frame for MENU
menu_frame = tk.LabelFrame(root)
menu_frame.place(height=250, width=700, rely=0.65, relx=0)
image = Image.open("background.jpg")
render = ImageTk.PhotoImage(image)
img = tk.Label(menu_frame, image=render)
img.image = render
img.place(x=0, y=0)

# Buttons
global button_id
button1 = tk.Button(menu_frame, text="Spam Classification v1", command=lambda: spam_classificationLOAD(), height=2,
                    width=25)
button1.place(rely=0.20, relx=0.1)
button1['background'] = '#BEBEBE'
CreateToolTip(button1, text='This technique\n'
                            'uses Random Forest Classifier')

button2 = tk.Button(menu_frame, text="Analyse", command=lambda: analyse(), height=2, width=15)
button2.place(rely=0.65, relx=0.35)
button2['background'] = '#BEBEBE'

button3 = tk.Button(menu_frame, text="Review Classification (Good/Bad)", command=lambda: review_predictionLOAD(), height=2,
                    width=25)
button3.place(rely=0.40, relx=0.1)
button3['background'] = '#BEBEBE'
CreateToolTip(button3, text='This technique\n'
                            'uses Linear SVC')

button4 = tk.Button(menu_frame, text="Spam Classification v2", command=lambda: spam_classificationCNN_LOAD(), height=2, width=25)
button4.place(rely=0.20, relx=0.5)
button4['background'] = '#BEBEBE'
CreateToolTip(button4, text='This technique\n'
                            'uses CNN')

button5 = tk.Button(menu_frame, text="Text Generation", command=lambda: UNKNOWNFUNCTIONLOAD2(), height=2, width=25)
button5.place(rely=0.40, relx=0.5)
button5['background'] = '#BEBEBE'

# The file/file path text
label_file = ttk.Label(menu_frame, text="No File Selected")
label_file.place(rely=0, relx=0)
label_file['background'] = '#BEBEBE'

# General Treeview Settings
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1)  # set the height and width of the widget to 100% of its container (frame1).
treescrolly = tk.Scrollbar(frame1, orient="vertical",
                           command=tv1.yview)  # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal",
                           command=tv1.xview)  # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set,
              yscrollcommand=treescrolly.set)  # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x")  # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y")  # make the scrollbar fill the y axis of the Treeview widget


# supports .tsv file
def spam_classificationLOAD():
    global button_id
    button_id = 1
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetypes=[("txt files", "*.txt"), ("tsv files", "*.tsv")])
    PATH = label_file["text"] = filename
    NLP.NLP_SpamDetection.locatePATH(PATH)
    return None


def review_predictionLOAD():
    global button_id
    button_id = 2
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetypes=[("txt files", "*.txt"), ("tsv files", "*.tsv")])
    try:
        PATH = label_file["text"] = filename
        NLP.NLP_RestaurantReviewClassification2.locatePATH(PATH)

    except ValueError:
        tk.messagebox.showerror("Information", "The file is Invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", "No such file")
        return None

    return None


def spam_classificationCNN_LOAD():
    global button_id
    button_id = 3
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetypes=[("txt files", "*.txt"), ("tsv files", "*.tsv")])
    PATH = label_file["text"] = filename
    NLP.NLP_SpamDetectionCNN.locatePATH(PATH)
    return None


def UNKNOWNFUNCTIONLOAD2():
    return None


def analyseSPAM():
    accuracy_score, results, predictions = NLP.NLP_SpamDetection.SimpleSpamDetection()
    tk.messagebox.showinfo("Info", f"Model Accuracy score: {accuracy_score}")

    # Create the result info
    tv1["columns"] = ("#1", '#2')
    tv1.heading('#1', text='Message')
    tv1.heading('#2', text='Prediction')
    tv1.column("#1", width=425)
    tv1.column("#2", width=150)

    tv1['show'] = 'headings'
    for i in range(0, len(results)):
        tv1.insert('', 0, values=(results[i], predictions[i]))


def analyseSPAM_CNN():
    accuracy_score, results, predictions = NLP.NLP_SpamDetectionCNN.SpamDectectionCNN()
    tk.messagebox.showinfo("Info", f"Model Accuracy score: {accuracy_score}")

    # Create the result info
    tv1["columns"] = ("#1", '#2')
    tv1.heading('#1', text='Message')
    tv1.heading('#2', text='% to be spam')
    tv1.column("#1", width=425)
    tv1.column("#2", width=150)

    tv1['show'] = 'headings'
    for i in range(0, len(results)):
        tv1.insert('', 0, values=(results[i], predictions[i]))


def review_prediction():
    accuracy_score, results, predictions = NLP.NLP_RestaurantReviewClassification2.reviewClassification()
    tk.messagebox.showinfo("Info", f"Model Accuracy score: {accuracy_score}")

    # Create the result info
    tv1["columns"] = ("#1", '#2')
    tv1.heading('#1', text='Review')
    tv1.heading('#2', text='Sentiment')
    tv1.column("#1", width=425)
    tv1.column("#2", width=150)

    tv1['show'] = 'headings'
    for i in range(0, len(results)):
        tv1.insert('', 0, values=(results[i], predictions[i]))


def analyse():
    if button_id == 1:
        analyseSPAM()
    elif button_id == 2:
        review_prediction()
    elif button_id == 3:
        analyseSPAM_CNN()


def clear_data():
    tv1.delete(*tv1.get_children())
    return None


root.mainloop()
