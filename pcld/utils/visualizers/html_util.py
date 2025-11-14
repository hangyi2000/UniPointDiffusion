# -*- coding: utf-8 -*-


def to_html_frame(content):

    html_frame = f"""
    <html>
      <body>
        {content}
      </body>
    </html>
    """

    return html_frame


def to_single_row_table(caption: str, content: str):

    table_html = f"""
    <table border = "1">
        <caption>{caption}</caption>
        <tr>
            <td>{content}</td>
        </tr>
    </table>
    """

    return table_html


def to_double_row_table(caption: str, content1: str, content2: str):

    table_html = f"""
    <table border = "1">
        <caption>{caption}</caption>
        <tr>
            <td>{content1}</td>
            <td>{content2}</td>
        </tr>
    </table>
    """

    return table_html
