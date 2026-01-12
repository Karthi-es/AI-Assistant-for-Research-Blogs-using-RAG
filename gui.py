import streamlit as st


class AssistantGUI:
    def __init__(self, assistant):
        self.assistant = assistant
        self.messages = assistant.messages

    def get_response(self, user_input):
        return self.assistant.get_response(user_input)

    def render_messages(self):
        for message in self.messages:
            if message["role"] == "user":
                st.chat_message("human").markdown(message["content"])
            if message["role"] == "ai":
                st.chat_message("ai").markdown(message["content"])

    def render_user_input(self):
        disabled = self.assistant.vector_store is None
        prompt = "Add sources in the sidebar to enable chat" if disabled else "Ask about the research..."
        user_input = st.chat_input(prompt, key="input", disabled=disabled)

        if not user_input:
            return

        st.chat_message("human").markdown(user_input)

        response_generator = self.get_response(user_input)

        with st.chat_message("ai"):
            response = st.write_stream(response_generator)

        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "ai", "content": response})

        st.session_state.messages = self.messages

    def render(self):
        self.render_messages()
        self.render_user_input()