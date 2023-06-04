package com.example.demo.pojo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
@Data
@AllArgsConstructor
@NoArgsConstructor
public class QApair {
    private String question;
    private String answer;
    private String context;
    private int start;
    private int end;
    private String[] all_answers;       //所有可能的答案
    private int id;
    private boolean isBing = false;     //是否是bing结果
}
