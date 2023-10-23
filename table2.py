import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import time


def pre_process_image(img, save_in_file, morph_size=(8, 8)):
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre = cv2.threshold(pre, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]
        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)
    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    table_cells = [list(sorted(tb)) for tb in table_cells]
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))
    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []
    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]
    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]
    hor_lines = []
    ver_lines = []
    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))
    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))
    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))
    return hor_lines, ver_lines


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


if __name__ == '__main__':
    file = 'data/in.jpg'
    imgs = cv2.imread(file)
    new_size = (750, 1061)
    imgs = cv2.resize(imgs, new_size)
    imgs = cv2.convertScaleAbs(imgs, alpha=1.5, beta=10)

    aligned_img = imgs

    aligned_img_grey = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    aligned_img_threshold = cv2.threshold(aligned_img_grey, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    entity_roi = [
        [(35, 220), (190, 250), 'text', 'Bill To'],
        [(30, 240), (205, 330), 'text', 'Bill To Address'],
        [(250, 220), (400, 250), 'text', 'Ship To'],
        [(250, 240), (420, 330), 'text', 'Ship To Address'],
        [(600, 230), (700, 230), 'text', 'InVoice No'],
        [(600, 240), (700, 260), 'text', 'Invoice Date'],
        [(600, 250), (700, 290), 'text', 'PO Number'],
        [(600, 260), (700, 320), 'text', 'Due Date'],
        [(700, 540), (700, 600), 'text', 'Total']
    ]

    aligned_img_show = aligned_img_threshold.copy()
    aligned_img_mask = np.zeros_like(aligned_img_show)

    entity_type = []
    entity_name = []
    entity_value = []

    for roi in entity_roi:
        top_left_x = roi[0][0]
        top_left_y = roi[0][1]
        bottom_right_x = roi[1][0]
        bottom_right_y = roi[1][1]
        cv2.rectangle(aligned_img_mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255),
                      cv2.FILLED)
        aligned_img_show = cv2.addWeighted(aligned_img_show, 0.99, aligned_img_mask, 0.1, 0)

        img_cropped = aligned_img_threshold[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        cv2.imshow('Masked Image', img_cropped)
        cv2.waitKey(0)
        custom_config = r'--oem 3 --psm 6'
        ocr_output = pytesseract.image_to_string(img_cropped, config=custom_config, lang='eng')
        cleaned_output = os.linesep.join([s for s in ocr_output.splitlines() if s])
        font = cv2.FONT_HERSHEY_DUPLEX
        red_color = (0, 0, 255)
        cv2.putText(aligned_img_threshold, f'{cleaned_output}', (top_left_x, top_left_y - 30), font, 0.5, red_color, 1)

        entity_type.append(roi[2])
        entity_name.append(roi[3])
        entity_value.append(cleaned_output)

    img_cropped = imgs
    aligned_img_show = img_cropped.copy()
    aligned_img_mask = np.zeros_like(aligned_img_show)
    cv2.rectangle(aligned_img_mask, (50, 335), (700, 620), (0, 0, 255), cv2.FILLED)
    aligned_img_show = cv2.addWeighted(aligned_img_show, 0.99, aligned_img_mask, 0.1, 0)
    img = img_cropped[335:620, 50:700]
    pre_file = os.path.join("data", "pre.png")
    out_file = os.path.join("data", "out.png")
    pre_processed = pre_process_image(img, pre_file)
    text_boxes = find_text_boxes(pre_processed)
    cells = find_table_in_boxes(text_boxes)
    hor_lines, ver_lines = build_lines(cells)
    vis = img.copy()
    for line in hor_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1 - 20, y1), (x2, y2), (0, 0, 255), 1)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(out_file, vis)

    file = r'data/out.png'
    img = cv2.imread(file, 0)

    thresh, img_bin = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.waitKey(0)
    img_bin = 255 - img_bin
    plotting = plt.imshow(img_bin, cmap='gray')
    kernel_len = np.array(img).shape[1] // 100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    plotting = plt.imshow(image_1, cmap='gray')

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    plotting = plt.imshow(image_2, cmap='gray')

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    plotting = plt.imshow(bitnot, cmap='gray')
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 1000 and h < 500):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    plotting = plt.imshow(image, cmap='gray')
    row = []
    column = []
    j = 0

    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]
                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                        finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=2)

                    out = pytesseract.image_to_string(erosion)
                    if (len(out) == 0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner + " " + out
                outer.append(inner)

    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    json = dataframe.to_json()
    entity_type.append("json")
    entity_name.append("Description")
    entity_value.append(json)

    form_data = pd.DataFrame()
    form_data['Entity_Name'] = entity_name
    form_data['Entity_Value'] = entity_value
    form_data['Entity_Type'] = entity_type

    print(form_data.to_json(orient='values'))
    time.sleep(10)
    data_set = {'page': 'Request', 'message': f"{str(form_data.to_json(orient='values'))}", 'time': time.time()}

    cv2.waitKey(0)
