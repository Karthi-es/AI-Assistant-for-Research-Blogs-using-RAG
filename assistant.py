from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


class Assistant:
    def __init__(
        self,
        system_prompt,
        llm,
        message_history=None,
        vector_store=None,
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.messages = message_history or []
        self.vector_store = vector_store

        self.chain = self._get_conversation_chain()

    def get_response(self, user_input):
        return self.chain.stream(user_input)

    def _get_conversation_chain(self):
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("conversation_history"),
                ("human", "{user_input}"),
            ]
        )

        chain = (
            {
                "retrieved_context": self._context_runnable(),
                "user_input": RunnablePassthrough(),
                "conversation_history": lambda _: self.messages,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _context_runnable(self):
        if not self.vector_store:
            return RunnableLambda(
                lambda _: "No knowledge base is available yet. Please add blog URLs or PDF papers to provide grounding."
            )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        return retriever | RunnableLambda(self._format_docs)

    @staticmethod
    def _format_docs(documents):
        if not documents:
            return "The retriever did not return any passages for this question."

        return "\n\n".join(doc.page_content for doc in documents)