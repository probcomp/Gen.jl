---
layout: splash
title: Posts 
---

<br/>
# Gen Blog

<table class="table posts">
  {% for post in site.posts %}
    <tr><td class="list-group-item-action flex-column align-items-start">

        <div class="d-flex w-100 justify-content-between">
            <h3 class="mb-1"><a href="{{ post.url }}">{{ post.title }}</a></h3>
            <div><b>{{post.author}}</b> - {{ post.date | date_to_string }}</div>
        </div>
        <p class="mb-1">{{ post.summary}}</p>
        </td>
   </tr>

  {% endfor %}
</table>

