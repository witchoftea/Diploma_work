import gradio as gr
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.load_trained_models import load_lightfm_model

model, dataset, item_features, mappings = load_lightfm_model()
df = mappings['df']
user_id_map = mappings['user_id_map']
item_id_map = mappings['item_id_map']
inv_user_id_map = {v: k for k, v in user_id_map.items()}
inv_item_id_map = {v: k for k, v in item_id_map.items()}
book_id_to_title = mappings['book_id_to_title']
top_genres = mappings['top_genres']
tag_to_books = mappings['tag_to_books']


def get_books_read_by_user(user_id):
    if user_id not in user_id_map:
        return "<p>–</p>"
    books_read = df[df['user_id'] == user_id][['title', 'authors']].dropna().drop_duplicates()
    books_read.columns = ["Назва книги", "Автор"]

    html = "<h4>Прочитані книги</h4>"
    html += """
    <div style='max-height: 180px; overflow-y: auto; border: 1px solid #ccc; padding: 0;'>
    <table style='width:100%; font-size: 14px; border-collapse: collapse;'>
        <thead style='position: sticky; top: 0; background: #f9f9f9; z-index: 1;'>
            <tr><th style='text-align:left; padding: 6px 8px;'>Назва книги</th><th style='text-align:left; padding: 6px 8px;'>Автор</th></tr>
        </thead>
        <tbody>
    """
    for _, row in books_read.iterrows():
        html += f"<tr><td style='padding: 4px 8px;'>{row['Назва книги']}</td><td style='padding: 4px 8px;'>{row['Автор']}</td></tr>"
    html += "</tbody></table></div>"
    return html

def format_book_list(book_ids):
    books = df[df['book_id'].isin(book_ids)].drop_duplicates('book_id')
    result = []
    for _, row in books.iterrows():
        image_url = row.get('small_image_url') or row.get('image_url')
        if pd.notna(image_url):
            title = row['title']
            author = row['authors'] if pd.notna(row['authors']) else "Невідомий автор"
            title_author = f"{title}<br><i>{author}</i>"
            result.append([image_url, title_author])
    return result

def format_recommendations_html(recommendations):
    html = "<h4>Рекомендовані книги</h4>"
    html += "<div style='display:flex; flex-wrap:wrap; gap:10px;'>"
    for image_url, title_author in recommendations:
        html += f"""
        <div style='width: 120px; text-align: center; font-size: 12px;'>
            <div>{title_author}</div>
            <img src='{image_url}' style='width:100px; height:auto; border:1px solid #ccc; margin-top:4px;'/>
        </div>
        """
    html += "</div>"
    return html

def recommend_books_for_user(user_id, model, dataset, item_features, N=5):
    if user_id not in user_id_map:
        return []

    internal_uid = user_id_map[user_id]
    all_items = list(item_id_map.values())
    scores = model.predict(internal_uid, all_items, item_features=item_features)

    top_item_indices = np.argsort(-scores)[:N]
    top_book_ids = [inv_item_id_map[i] for i in top_item_indices]
    return format_book_list(top_book_ids)

def gradio_recommender_with_history(input_user_id):
    try:
        user_id = int(input_user_id)
        read = get_books_read_by_user(user_id)
        recommendations = recommend_books_for_user(user_id, model, dataset, item_features, N=5)
        html_rec = format_recommendations_html(recommendations)
        return read, html_rec
    except:
        return "<p>–</p>", "Введіть коректний числовий user_id."

def recommend_by_genre(selected_genres, top_n=5):
    selected_genres = [g.lower() for g in selected_genres]
    book_candidates = defaultdict(int)

    for genre in selected_genres:
        for book_id, count in tag_to_books.get(genre, []):
            if book_id in book_id_to_title:
                book_candidates[book_id] += count

    if not book_candidates:
        return "Нічого не знайдено."

    top_books = sorted(book_candidates.items(), key=lambda x: -x[1])[:top_n]
    top_book_ids = [bid for bid, _ in top_books]
    formatted = format_book_list(top_book_ids)
    return format_recommendations_html(formatted)

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Персоналізована система рекомендацій книг")

    with gr.Tab("Існуючий користувач"):
        user_input = gr.Textbox(label="User ID", placeholder="наприклад, 42")
        read_output = gr.HTML()
        rec_output = gr.HTML()
        recommend_btn = gr.Button("Отримати рекомендації")

        def full_recommend(user_id):
            read, rec = gradio_recommender_with_history(user_id)
            return read, rec

        recommend_btn.click(fn=full_recommend, inputs=user_input, outputs=[read_output, rec_output])

    with gr.Tab("Новий користувач"):
        genre_input = gr.CheckboxGroup(choices=top_genres, label="Оберіть жанри")
        genre_rec_output = gr.HTML()
        genre_btn = gr.Button("Підібрати")

        def wrapped_recommend_by_genre(genres):
            return recommend_by_genre(genres)

        genre_btn.click(fn=wrapped_recommend_by_genre, inputs=genre_input, outputs=genre_rec_output)

demo.launch()
