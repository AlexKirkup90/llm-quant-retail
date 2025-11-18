import streamlit as st

class SimpleView:
    def __init__(self, controller):
        self.controller = controller

    def render(self):
        st.title("Simplified Quant App")

        st.header("1) Analyse Market Conditions")
        if st.button("Run Market Analysis"):
            self.controller.analyse_market_conditions()

        st.header("2) Analyse Current Portfolio")
        if st.button("Run Portfolio Analysis"):
            self.controller.analyse_current_portfolio()

        st.header("3) Generate New Portfolio")
        if st.button("Generate New Portfolio"):
            self.controller.generate_new_portfolio()
