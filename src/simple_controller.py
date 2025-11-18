import streamlit as st

class SimpleController:
    def __init__(self, controller):
        self.controller = controller

    def analyse_market_conditions(self):
        # Delegate to the original controller to analyse market conditions
        self.controller.analyse_market_conditions()
        st.write("Market conditions analysed.")

    def analyse_current_portfolio(self):
        # Delegate to the original controller to analyse the current portfolio
        self.controller.analyse_current_portfolio()
        st.write("Current portfolio analysed.")

    def generate_new_portfolio(self):
        # Delegate to the original controller to generate a new portfolio
        self.controller.generate_new_portfolio()
        st.write("New portfolio generated.")
