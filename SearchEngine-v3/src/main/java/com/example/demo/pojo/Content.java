package com.example.demo.pojo;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Collections;
import java.util.Iterator;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Content {
    private String title;
//    private String img;
//    private String price;
    private String href;    // 搜索引擎连接
    private String caption;     //单条搜索结果的说明
    private String[] emphasis;  //搜索引擎单条结果下的强调，约等于all_answers
    //可以自行添加属性
    private String context;     //搜索结果链接解析后的文本

    public Iterator<Content> iterator() {
        return Collections.singletonList(this).iterator();
    }
}