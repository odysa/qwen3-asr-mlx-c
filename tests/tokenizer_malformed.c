#include "tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){assert(argc==2);FILE *f=fopen(argv[1],"wb");assert(f);const char *cases[]={"{\"x\\","{\"x\\u12","{\"x\\uZZZZ\":1}","{\"x\":99999999999999999999999999999999999999999999999999}","{\"x\":-1}","{\"x\":2000001}","{\"x\":x}","{\"x\":1","{}","{\"x\\q\":1}","{\"x\\uD800\":1}"};fclose(f);for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++){f=fopen(argv[1],"wb");assert(f);fwrite(cases[i],1,strlen(cases[i]),f);fclose(f);qwen_tokenizer_t *t=qwen_tokenizer_load(argv[1]);if(t){qwen_tokenizer_free(t);fprintf(stderr,"accepted malformed case%zu\n",i);return 1;}}f=fopen(argv[1],"wb");assert(f);fputs("{\"a\":0}",f);fclose(f);qwen_tokenizer_t *good=qwen_tokenizer_load(argv[1]);assert(good);qwen_tokenizer_free(good);remove(argv[1]);puts("11 malformed vocab cases rejected");return 0;}
