            drift_df,
            use_container_width=True,
        )

        fig = px.bar(
            drift_df.head(20),
            x="Drift %",
            y="Feature",
            orientation="h",
            title="Top Feature Distribution Changes",
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#111827",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        "This monitoring page provides a simple analytical drift indicator "
        "by comparing feature means. It is intended for project monitoring, "
        "not production regulatory monitoring."
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📋 Dataset Explorer":

    st.title(
        "📋 Dataset Explorer"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{df.shape[0]:,}",
    )

    c2.metric(
        "Columns",
        df.shape[1],
    )

    c3.metric(
        "Duplicates",
        f"{df.duplicated().sum():,}",
    )

    c4.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}",
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.subheader(
        "Column Information"
    )

    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing": [
                df[c].isnull().sum()
                for c in df.columns
            ],
            "Unique": [
                df[c].nunique()
                for c in df.columns
            ],
        }
    )

    st.dataframe(
        info,
        use_container_width=True,
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe(
            include="all"
        ).transpose(),
        use_container_width=True,
    )

    st.download_button(
        "📥 Download Dataset",
        df.to_csv(index=False),
        "financial_fraud_dataset.csv",
        "text/csv",
    )
